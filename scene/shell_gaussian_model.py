import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from flame_model.flame import FlameHead
from utils.network_utils import DynamicDecoder
from pathlib import Path
from .gaussian_model import GaussianModel
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.graphics_utils import compute_face_orientation
from pytorch3d.structures import Meshes
from torch.autograd.functional import jacobian
from torch.nn import functional as F
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
import cv2

class ShellGaussianModel(GaussianModel):
    
    def setup_functions(self):
        super().setup_functions()

        self.bc_activation = torch.exp
        self.bc_inverse_activation = torch.log
    
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, not_finetune_xyz=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.not_finetune_xyz = not_finetune_xyz
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None
        self.timestep = 0  # TODO：改所有帧/单帧

        self.verts_uvs = self.flame_model.verts_uvs.unsqueeze(0).unsqueeze(2) # (1, 16428, 1, 2)
        self.split_verts_uvs = self.flame_model.split_verts_uvs # (16428, 2)

        # self.face_num is initialized once the mesh topology is known
        self.current_num_shells = 1
        self.num_shells = 2
        self.init_num_points = 200_000
        self.add_num_points = 600_000
        self.num_points = self.init_num_points + self.add_num_points
        
        self.bc = torch.empty(0)
        self.bc_gradient_accum = torch.empty(0)
        self._xyz = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        
        self.new_textures_dc = None
        self._textures_dc = torch.empty(0)
        self._textures_expr = torch.empty(0)

        self.texture_size = 1024

        indice_face = self.flame_model._parts['face']
        indice_lips = self.flame_model._parts['lips']
        self.indice_face = indice_face[~torch.isin(indice_face, indice_lips)].cuda()

        num_points = self.flame_model.v_template.shape[0]
        num_faces = self.flame_model.faces.shape[0]
        
        if self.face_num is None:
            def create_face_counter(num_points, num_faces):
                points_per_face = num_points // num_faces
                extra_points = num_points % num_faces
                face_counter = torch.ones(num_faces, dtype=torch.int32, device="cuda") * points_per_face
                if extra_points > 0:
                    indices = torch.randperm(num_faces)[:extra_points]
                    face_counter[indices] += 1
                face_num = torch.arange(num_faces).cuda().repeat_interleave(face_counter)
                return face_counter, face_num
            init_face_counter, init_face_num = create_face_counter(self.init_num_points, num_faces)
            _, add_face_num = create_face_counter(self.add_num_points, num_faces)
            self.face_num = [init_face_num, add_face_num]
            self.init_face_counter = init_face_counter

        self.split_faces = self.flame_model.split_faces

        '''adjacent_faces = self.flame_model.adjacent_faces.repeat(self.num_shells, 1) + offsets
        self.adjacent_faces = adjacent_faces
        self.roll_num = self.flame_model.roll_num.repeat(self.num_shells, 1)
        self.if_flip = self.flame_model.if_flip.repeat(self.num_shells, 1)'''

        self.face_mask = (torch.isin(self.flame_model.faces, self.indice_face).all(dim=1)).repeat(self.num_shells, 1).reshape(-1)

        self.setup_functions()
    
    def capture(self):  # TODO：完成
        raise NotImplementedError

    def restore(self):  # TODO：完成
        raise NotImplementedError
    
    def create_from_mesh(self, spatial_lr_scale : float):
        # xyz initialization
        self.spatial_lr_scale = spatial_lr_scale
        self.bc = nn.Parameter(self.bc_inverse_activation(torch.rand(self.init_num_points, 3).cuda())).requires_grad_(False)
        self._xyz = nn.Parameter(torch.zeros((self.add_num_points, 3), dtype=torch.float, device="cuda")).requires_grad_(False)
        
        # features initialization
        self._features_dc = [nn.Parameter(torch.zeros((self.init_num_points, 1, 3), dtype=torch.float, device="cuda")).requires_grad_(False),
                             nn.Parameter(torch.zeros((self.add_num_points, 1, 3), dtype=torch.float, device="cuda")).requires_grad_(False)]
        self._features_rest = [nn.Parameter(torch.zeros((self.init_num_points, (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float, device="cuda")).contiguous().requires_grad_(False),
                               nn.Parameter(torch.zeros((self.add_num_points, (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float, device="cuda")).contiguous().requires_grad_(False)]

        self._textures_dc = nn.Parameter(torch.zeros((1, 3, self.texture_size, self.texture_size), dtype=torch.float, device="cuda")).requires_grad_(False)
        self._textures_expr = DynamicDecoder().cuda()
        
        # scaling, rotation, opacity initialization
        self._scaling = [nn.Parameter(torch.log(torch.ones((self.init_num_points, 3), dtype=torch.float, device="cuda"))).requires_grad_(False),
                         nn.Parameter(torch.log(torch.ones((self.add_num_points, 3), dtype=torch.float, device="cuda"))).requires_grad_(False)]
        
        rots_0 = torch.zeros((self.init_num_points, 4), device="cuda")
        rots_0[:, 0] = 1
        rots_1 = torch.zeros((self.add_num_points, 4), device="cuda")
        rots_1[:, 0] = 1
        self._rotation = [nn.Parameter(rots_0).requires_grad_(False),
                          nn.Parameter(rots_1).requires_grad_(False)]

        self._opacity = [nn.Parameter(inverse_sigmoid(0.1 * torch.ones((self.init_num_points, 1), dtype=torch.float, device="cuda"))).requires_grad_(False),
                         nn.Parameter(inverse_sigmoid(0.1 * torch.ones((self.add_num_points, 1), dtype=torch.float, device="cuda"))).requires_grad_(False)]

        self.max_radii2D = [torch.zeros((self.init_num_points), device="cuda"),
                            torch.zeros((self.add_num_points), device="cuda")]

        print("Number of points at initialisation: ", self.num_points)
    
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])   # TODO：改所有帧/单帧
                static_offset = F.pad(static_offset, (0, 0, 0, 5023 - static_offset.shape[1]))
                new_static_offset = static_offset.squeeze(0)[self.flame_model.subdivide_edges]
                new_static_offset = new_static_offset.mean(dim=1).unsqueeze(0)
                static_offset = torch.cat((static_offset, new_static_offset), dim=1)
                if static_offset.shape[0] != num_verts:
                    static_offset = F.pad(static_offset, (0, 0, 0, num_verts - static_offset.shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                # 'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.denom = [torch.zeros((self.bc.shape[0], 1), device="cuda"),
                      torch.zeros((self._xyz.shape[0], 1), device="cuda")]
        self._textures_dc.requires_grad = True
        self._features_rest[0].requires_grad = True
        self._scaling[0].requires_grad = True
        self._opacity[0].requires_grad = True
        
        param_base = [
            {'params': [self._textures_dc], 'lr': training_args.f_dc_lr, "name": "t_dc_0"},
            {'params': [self._features_rest[0]], 'lr': training_args.f_rest_lr, "name": "f_rest_0"},
            {'params': [self._opacity[0]], 'lr': training_args.opacity_lr, "name": "opacity_0"},
            {'params': [self._scaling[0]], 'lr': training_args.scaling_lr, "name": "scaling_0"},
        ]

        self.optimizer = torch.optim.Adam(param_base, lr=0.0, eps=1e-15)
        
        self.bc_gradient_accum = torch.zeros((self.bc.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        param_t_expr = {'params': self._textures_expr.parameters(), 'lr': training_args.f_dc_lr / 10.0, "name": "t_expr"}
        self.optimizer.add_param_group(param_t_expr)
        
        if self.not_finetune_xyz == False:
            self.bc.requires_grad = True
            param_bc = {'params': [self.bc], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "bc_0"}
            self.optimizer.add_param_group(param_bc)
            self.bc_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.bc_position_lr_max_steps)
        
        if self.not_finetune_flame_params == False:
            # # shape
            # self.flame_param['shape'].requires_grad = True
            # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
            # self.optimizer.add_param_group(param_shape)

            # pose
            self.flame_param['rotation'].requires_grad = True
            self.flame_param['neck_pose'].requires_grad = True
            self.flame_param['jaw_pose'].requires_grad = True
            self.flame_param['eyes_pose'].requires_grad = True
            params = [
                self.flame_param['rotation'],
                self.flame_param['neck_pose'],
                self.flame_param['jaw_pose'],
                self.flame_param['eyes_pose'],
            ]
            param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
            self.optimizer.add_param_group(param_pose)

            # translation
            self.flame_param['translation'].requires_grad = True
            param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
            self.optimizer.add_param_group(param_trans)
            
            # expression
            self.flame_param['expr'].requires_grad = True
            param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
            self.optimizer.add_param_group(param_expr)
    
    def set_stage(self, stage, training_args):
        if stage == 0:
            self._textures_dc.requires_grad = True
            for param in self._textures_expr.parameters():
                param.requires_grad = True
            self._features_dc[0].requires_grad = False
            self._features_dc[1].requires_grad = False
            self._features_rest[0].requires_grad = True
            self._features_rest[1].requires_grad = False
            self._scaling[0].requires_grad = True
            self._scaling[1].requires_grad = False
            self._opacity[0].requires_grad = True
            self._opacity[1].requires_grad = False
            self._rotation[0].requires_grad = False
            self._rotation[1].requires_grad = False
            if self.not_finetune_xyz == False:
                self._xyz.requires_grad = False
                self.bc.requires_grad = True
            
            self.current_num_shells = 1
        
        elif stage == 1:
            self.add_gaussians(training_args)

            self.current_num_shells = 2
    
    def add_gaussians(self, training_args):
        self.current_num_shells += 1

        self._textures_dc.requires_grad = False
        self._textures_expr.requires_grad = False
        self._features_dc[0].requires_grad = False
        self._features_rest[0].requires_grad = False
        self._scaling[0].requires_grad = False
        self._opacity[0].requires_grad = False
        # TODO：init from iteration时改成True，不然reset会错
        self._rotation[0].requires_grad = False
        
        self._features_dc[1].requires_grad = True
        self._features_rest[1].requires_grad = True
        self._scaling[1].requires_grad = True
        self._opacity[1].requires_grad = True
        self._rotation[1].requires_grad = True
        
        param_t_dc_1 = {'params': [self._features_dc[1]], 'lr': training_args.f_dc_lr, "name": "f_dc_1"}
        param_f_rest_1 = {'params': [self._features_rest[1]], 'lr': training_args.f_rest_lr, "name": "f_rest_1"}
        param_scaling_1 = {'params': [self._scaling[1]], 'lr': training_args.scaling_lr, "name": "scaling_1"}
        param_opacity_1 = {'params': [self._opacity[1]], 'lr': training_args.opacity_lr, "name": "opacity_1"}
        param_rotation_1 = {'params': [self._rotation[1]], 'lr': training_args.rotation_lr, "name": "rotation_1"}
        
        self.optimizer.add_param_group(param_t_dc_1)
        self.optimizer.add_param_group(param_f_rest_1)
        self.optimizer.add_param_group(param_scaling_1)
        self.optimizer.add_param_group(param_opacity_1)
        self.optimizer.add_param_group(param_rotation_1)
        
        if self.not_finetune_xyz == False:
            self._xyz.requires_grad = True
            self.bc.requires_grad = False
            param_xyz = {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_1"}
            self.optimizer.add_param_group(param_xyz)
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.xyz_position_lr_max_steps)
    
    def update_learning_rate(self, iteration):
        # Learning rate scheduling per step
        for param_group in self.optimizer.param_groups:
            if "bc" in param_group["name"]:
                lr = self.bc_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif "xyz" in param_group["name"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    @property
    def get_xyz(self):
        if self.face_center is None:
            self.select_mesh_by_timestep(0)
        
        verts = (self.verts.squeeze(0))[self.faces[self.face_num[0]]]
        xyz_0 = (verts * self.get_bc[..., None]).sum(dim=1)
        
        xyz_1 = torch.bmm(self.face_orien_mat[self.face_num[1]], self._xyz[..., None]).squeeze(-1)
        xyz_1 = xyz_1 * self.face_scaling[self.face_num[1]] + self.face_center[self.face_num[1]]

        return torch.cat((xyz_0, xyz_1), dim=0)

    @property
    def get_combined_scaling(self):
        return torch.cat(self._scaling, dim=0)
    
    @property
    def get_scaling(self):
        if self.face_scaling is None:
            self.select_mesh_by_timestep(0)

        scaling = self.scaling_activation(self.get_combined_scaling)
        return scaling * self.face_scaling[self.get_face_num]
    
    @property
    def get_combined_rotation(self):
        return torch.cat(self._rotation, dim=0)

    @property
    def get_rotation(self):
        if self.face_orien_quat is None:
            self.select_mesh_by_timestep(0)

        # always need to normalize the rotation quaternions before chaining them
        rot = self.rotation_activation(self.get_combined_rotation)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[self.get_face_num])
        return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))  # roma
        # return quaternion_multiply(face_orien_quat, rot)  # pytorch3d
    
    @property
    def get_opacity(self):
        processed_opacities = []

        for idx, opacity_tensor in enumerate(self._opacity):
            if idx < self.current_num_shells:
                processed_opacities.append(self.opacity_activation(opacity_tensor))
            else:
                processed_opacities.append(torch.zeros_like(opacity_tensor))

        return torch.cat(processed_opacities, dim=0)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_combined_rotation)
    
    @property
    def get_bc(self):
        bc_tmp = self.bc_activation(self.bc)
        bc_tmp = bc_tmp / bc_tmp.sum(dim=1, keepdim=True)
        return bc_tmp
    
    @property
    def get_gradient_accum(self):
        return torch.cat([self.bc_gradient_accum, self.xyz_gradient_accum], dim=0)
    
    @property
    def get_denom(self):
        return torch.cat(self.denom, dim=0)
    
    @property
    def get_max_radii2D(self):
        return torch.cat(self.max_radii2D, dim=0)
    
    def get_textures(self, expr):
        return (self.get_textures_dc + self.get_textures_expr if expr else self.get_textures_dc).squeeze(0).permute(1, 2, 0)

    def get_raw_textures(self, expr):
        return self.get_textures_dc + self.get_textures_expr if expr else self.get_textures_dc
    
    @property
    def get_textures_dc(self):
        if self.new_textures_dc != None:
            return self.new_textures_dc
        else:
            return self._textures_dc

    @property
    def get_textures_expr(self):
        expr = self.flame_param['expr'][[self.timestep]].clone().detach()
        texture = self._textures_expr(expr)
        if self.texture_size != texture.shape[-1]:
            return F.interpolate(texture, size=(self.texture_size, self.texture_size), mode='bilinear', align_corners=True)
        else:
            return texture
    
    @property
    def get_features(self):
        features_dc = self.get_features_dc
        features_rest = self.get_features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        f_dc = [_ for _ in self._features_dc]
        f_dc[0] = nn.functional.grid_sample(self.get_raw_textures(expr=True), self.get_uvs.unsqueeze(0).unsqueeze(1), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(0).transpose(0, 2).detach().requires_grad_(False)
        return torch.cat(self._features_dc, dim=0)
    
    @property
    def get_features_rest(self):
        return torch.cat(self._features_rest, dim=0)

    @property
    def get_uvs(self):
        middle_uvs = self.split_verts_uvs[self.split_faces[self.face_num[0]]]
        self.points_uvs = (middle_uvs * self.get_bc[..., None]).sum(dim=1)
        return self.points_uvs
    
    @property
    def get_grad_uvs(self):
        def bc2puv(inputs):
            uvs = self.split_verts_uvs[self.split_faces[self.face_num[0]]].detach().requires_grad_(False)
            return (uvs * inputs[..., None]).sum(dim=1).sum(dim=0)
        def bc2xyz(inputs):
            verts = (self.verts.squeeze(0))[self.faces[self.face_num[0]]].detach().requires_grad_(False)
            return (verts * inputs[..., None]).sum(dim=1).sum(dim=0)
        bc = self.get_bc.clone().detach()
        grad_bc2puv = jacobian(func=bc2puv, inputs=bc)  # 形状：[2, self.init_num_points, 3]
        grad_bc2xyz = jacobian(func=bc2xyz, inputs=bc)  # 形状：[3, self.init_num_points, 3]
        return torch.matmul(grad_bc2puv.transpose(0, 1), torch.linalg.inv(grad_bc2xyz.transpose(0, 1))).reshape(-1, 6).contiguous().detach().requires_grad_(False)
    
    @property
    def get_face_num(self):
        return torch.cat(self.face_num, dim=0)
    
    def write_obj(self, filepath, verts, tris=None, log=True):
        """将mesh顶点与三角面片存储为.obj文件,方便查看

        Args:
            verts:      Vx3, vertices coordinates
            tris:       n_facex3, faces consisting of vertices id
        """
        fw = open(filepath, "w")
        # vertices
        for vert in verts:
            fw.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        if not tris is None:
            for tri in tris:
                fw.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        fw.close()
        if log:
            print(f"mesh has been saved in {filepath}.")
        
    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=None,    # TODO：加上
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        # self.verts_cano = verts_cano
        # TODO：是否需要加上dynamic offset？
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))

        start_idx = self.calculate_idx()
        for i in range(0, self.current_num_shells):
            opacities_new_layer = opacities_new[start_idx[i]:start_idx[i+1]]
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new_layer, "opacity_"+str(i))
            self._opacity[i] = optimizable_tensors["opacity_"+str(i)]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, i, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if len(group["params"]) != 1 or group["params"][0].shape[0] != mask.shape[0] or "_"+str(i) not in group["name"]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, current_layer, mask):
        if current_layer == 0:
            face_num_to_prune = self.face_num[current_layer][mask] % self.init_face_counter.shape[0]
            counter_prune = torch.zeros_like(self.init_face_counter)
            counter_prune.scatter_add_(0, face_num_to_prune, torch.ones_like(face_num_to_prune, dtype=torch.int32, device="cuda"))
            mask_redundant = (self.init_face_counter - counter_prune) > 0
            mask[mask.clone()] = mask_redundant[face_num_to_prune]

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(current_layer, valid_points_mask)

        if current_layer == 0:
            self.bc = optimizable_tensors["bc_"+str(current_layer)]
        else:
            self._xyz = optimizable_tensors["xyz_"+str(current_layer)]
        self._features_rest[current_layer] = optimizable_tensors["f_rest_"+str(current_layer)]
        self._opacity[current_layer] = optimizable_tensors["opacity_"+str(current_layer)]
        self._scaling[current_layer] = optimizable_tensors["scaling_"+str(current_layer)]
        if current_layer == 0:
            self.bc = optimizable_tensors["bc_0"]
            self._rotation[current_layer] = self._rotation[current_layer][valid_points_mask]
            self._features_dc[current_layer] = self._features_dc[current_layer][valid_points_mask]
            self.bc_gradient_accum = self.bc_gradient_accum[valid_points_mask]
        else:
            self._rotation[current_layer] = optimizable_tensors["rotation_"+str(current_layer)]
            self._features_dc[current_layer] = optimizable_tensors["f_dc_"+str(current_layer)]
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom[current_layer] = self.denom[current_layer][valid_points_mask]
        self.max_radii2D[current_layer] = self.max_radii2D[current_layer][valid_points_mask]

        if current_layer == 0:
            face_num_to_prune = self.face_num[current_layer][mask] % self.init_face_counter.shape[0]
            self.init_face_counter.scatter_add_(0, face_num_to_prune, -torch.ones_like(face_num_to_prune, dtype=torch.int32, device="cuda"))
        self.face_num[current_layer] = self.face_num[current_layer][valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if group["name"] not in tensors_dict:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将新的密集化点的相关特征保存在一个字典中。
    def densification_postfix(self, current_layer, new_bc, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        if new_rotation is not None and new_features_dc is not None:
            if current_layer == 0:
                d = {"bc_0": new_bc,
                    "f_dc_"+str(current_layer): new_features_dc,
                    "f_rest_"+str(current_layer): new_features_rest,
                    "opacity_"+str(current_layer): new_opacities,
                    "scaling_"+str(current_layer): new_scaling,
                    "rotation_"+str(current_layer): new_rotation}
            else:
                d = {"xyz_1": new_bc,
                    "f_dc_"+str(current_layer): new_features_dc,
                    "f_rest_"+str(current_layer): new_features_rest,
                    "opacity_"+str(current_layer): new_opacities,
                    "scaling_"+str(current_layer): new_scaling,
                    "rotation_"+str(current_layer): new_rotation}
        elif new_rotation is None and new_features_dc is None:
            if current_layer == 0:
                d = {"bc_0": new_bc,
                    "f_rest_"+str(current_layer): new_features_rest,
                    "opacity_"+str(current_layer): new_opacities,
                    "scaling_"+str(current_layer): new_scaling}
            else:
                d = {"xyz_1": new_bc,
                    "f_rest_"+str(current_layer): new_features_rest,
                    "opacity_"+str(current_layer): new_opacities,
                    "scaling_"+str(current_layer): new_scaling}
        else:
            raise ValueError("new_features_dc and new_rotation must be both None or both not None.")

        # 将字典中的张量连接（concatenate）成可优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便于在优化器中进行处理。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if current_layer == 0:
            self.bc = optimizable_tensors["bc_0"]
            self.bc_gradient_accum = torch.zeros((self.bc.shape[0], 1), device="cuda")
            self.denom[current_layer] = torch.zeros((self.bc.shape[0], 1), device="cuda")
            self.max_radii2D[current_layer] = torch.zeros((self.bc.shape[0]), device="cuda")
        else:
            self._xyz = optimizable_tensors["xyz_1"]
            self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
            self.denom[current_layer] = torch.zeros((self._xyz.shape[0], 1), device="cuda")
            self.max_radii2D[current_layer] = torch.zeros((self._xyz.shape[0]), device="cuda")
        self._features_rest[current_layer] = optimizable_tensors["f_rest_"+str(current_layer)]
        self._opacity[current_layer] = optimizable_tensors["opacity_"+str(current_layer)]
        self._scaling[current_layer] = optimizable_tensors["scaling_"+str(current_layer)]
        if new_rotation is not None:
            self._rotation[current_layer] = optimizable_tensors["rotation_"+str(current_layer)]
        if new_features_dc is not None:
            self._features_dc[current_layer] = optimizable_tensors["f_dc_"+str(current_layer)]

    def densify_and_split(self, grads, grad_threshold, scene_extent, current_layer, N=2):
        n_init_points = self._xyz.shape[0] + self.bc.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        start_idx = self.calculate_idx()

        selected_pts_mask_layer = selected_pts_mask[start_idx[current_layer]:start_idx[current_layer+1]]
        if selected_pts_mask_layer.sum() == 0:
            return
        stds = (self.get_scaling[start_idx[current_layer]:start_idx[current_layer+1]])[selected_pts_mask_layer].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[current_layer][selected_pts_mask_layer]).repeat(N,1,1)
        if current_layer == 0:
            new_bc = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.bc[selected_pts_mask_layer].repeat(N, 1)
        else:
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) # + self._xyz[selected_pts_mask_layer].repeat(N, 1)
            # TODO：加上
        selected_scaling = (self.get_scaling[start_idx[current_layer]:start_idx[current_layer+1]])[selected_pts_mask_layer]
        face_scaling = self.face_scaling[self.face_num[current_layer][selected_pts_mask_layer]]
        new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[current_layer][selected_pts_mask_layer].repeat(N,1)
        new_features_dc = self._features_dc[current_layer][selected_pts_mask_layer].repeat(N,1,1)
        new_features_rest = self._features_rest[current_layer][selected_pts_mask_layer].repeat(N,1,1)
        new_opacity = self._opacity[current_layer][selected_pts_mask_layer].repeat(N,1)
        new_face_num = self.face_num[current_layer][selected_pts_mask_layer].repeat(N)
        self.face_num[current_layer] = torch.cat((self.face_num[current_layer], new_face_num))
        if current_layer == 0:
            new_face_num = new_face_num % self.init_face_counter.shape[0]
            self.init_face_counter.scatter_add_(0, new_face_num, torch.ones_like(new_face_num, dtype=torch.int32, device="cuda"))
            self._features_dc[current_layer] = torch.cat((self._features_dc[current_layer], new_features_dc))
            self._rotation[current_layer] = torch.cat((self._rotation[current_layer], new_rotation))
            self.densification_postfix(current_layer, new_bc, None, new_features_rest, new_opacity, new_scaling, None)
        else:
            self.densification_postfix(current_layer, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        prune_filter = torch.cat((selected_pts_mask_layer, torch.zeros(N * selected_pts_mask_layer.sum(), device="cuda", dtype=bool)))
        self.prune_points(current_layer, prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, current_layer):
        # Extract points that satisfy the gradient condition
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        start_idx = self.calculate_idx()

        selected_pts_mask_layer = selected_pts_mask[start_idx[current_layer]:start_idx[current_layer+1]]
        if selected_pts_mask_layer.sum() == 0:
            return
        new_features_dc = self._features_dc[current_layer][selected_pts_mask_layer]
        new_features_rest = self._features_rest[current_layer][selected_pts_mask_layer]
        new_opacities = self._opacity[current_layer][selected_pts_mask_layer]
        new_scaling = self._scaling[current_layer][selected_pts_mask_layer]
        new_rotation = self._rotation[current_layer][selected_pts_mask_layer]
        new_face_num = self.face_num[current_layer][selected_pts_mask_layer]
        self.face_num[current_layer] = torch.cat((self.face_num[current_layer], new_face_num))
        if current_layer == 0:
            new_bc = self.bc[selected_pts_mask_layer]
            new_face_num = new_face_num % self.init_face_counter.shape[0]
            self.init_face_counter.scatter_add_(0, new_face_num, torch.ones_like(new_face_num, dtype=torch.int32, device="cuda"))
            self._features_dc[current_layer] = torch.cat((self._features_dc[current_layer], new_features_dc))
            self._rotation[current_layer] = torch.cat((self._rotation[current_layer], new_rotation))
            self.densification_postfix(current_layer, new_bc, None, new_features_rest, new_opacities, new_scaling, None)
        else:
            new_xyz = self._xyz[selected_pts_mask_layer]
            self.densification_postfix(current_layer, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, current_layer):
        grads = self.get_gradient_accum / self.get_denom     # 计算密度估计的梯度
        grads[grads.isnan()] = 0.0  # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性。

        self.densify_and_clone(grads, max_grad, extent, current_layer)     # 对 under reconstruction 的区域进行稠密化和复制操作
        self.densify_and_split(grads, max_grad, extent, current_layer)     # 对 over reconstruction 的区域进行稠密化和分割操作

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.get_max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        start_idx = self.calculate_idx()
        prune_mask_layer = prune_mask[start_idx[current_layer]:start_idx[current_layer+1]]
        if prune_mask_layer.sum() == 0:
            return
        self.prune_points(current_layer, prune_mask_layer)

        torch.cuda.empty_cache()

    def calculate_idx(self):
        return torch.cumsum(torch.tensor([0, self.bc.shape[0], self._xyz.shape[0]]), dim=0)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        start_idx = self.calculate_idx()
        for i in range(0, self.current_num_shells):
            update_filter_layer = update_filter[start_idx[i]:start_idx[i+1]]
            if update_filter_layer.sum() == 0:
                continue
            if i == 0:
                self.bc_gradient_accum[update_filter_layer] += torch.norm(viewspace_point_tensor.grad[start_idx[i]:start_idx[i+1]][update_filter_layer,:2], dim=-1, keepdim=True)
            else:
                self.xyz_gradient_accum[update_filter_layer] += torch.norm(viewspace_point_tensor.grad[start_idx[i]:start_idx[i+1]][update_filter_layer,:2], dim=-1, keepdim=True)
            self.denom[i][update_filter_layer] += 1
    
    def save_params(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint = {
            'bc': self.bc.detach().cpu().numpy(),
            'xyz': self._xyz.detach().cpu().numpy(),
            't_expr': self._textures_expr.state_dict(),     # 保存卷积核
            't_dc': self._textures_dc.detach().cpu().numpy(),
            'f_dc': [_.detach().cpu().numpy() for _ in self._features_dc],
            'f_rest': [_.detach().cpu().numpy() for _ in self._features_rest],
            'opacity': [_.detach().cpu().numpy() for _ in self._opacity],
            'scaling': [_.detach().cpu().numpy() for _ in self._scaling],
            'rotation': [_.detach().cpu().numpy() for _ in self._rotation],
            'face_num': [_.detach().cpu().numpy() for _ in self.face_num],
            'init_face_counter': self.init_face_counter.detach().cpu().numpy(),
            'texture_size': self.texture_size,
            'max_radii2D': [_.detach().cpu().numpy() for _ in self.max_radii2D],
            'num_shells': self.num_shells,
            'current_num_shells': self.current_num_shells,
        }

        torch.save(checkpoint, path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)
        
        print(f"Checkpoint saved to {path}")
    
    def load_params(self, path, **kwargs):
        checkpoint = torch.load(path)

        self._textures_expr = DynamicDecoder().cuda()
        self.texture_size = checkpoint['texture_size']
        self.num_shells = checkpoint['num_shells']
        self.current_num_shells = checkpoint['current_num_shells']
        self.max_radii2D = [torch.tensor(_, dtype=torch.float, device="cuda") for _ in checkpoint['max_radii2D']]

        self.bc = nn.Parameter(torch.tensor(checkpoint['bc'], dtype=torch.float, device="cuda")).requires_grad_(False)
        self._xyz = nn.Parameter(torch.tensor(checkpoint['xyz'], dtype=torch.float, device="cuda")).requires_grad_(False)
        # load textures_dc
        self._textures_dc = nn.Parameter(torch.tensor(checkpoint['t_dc'], dtype=torch.float, device="cuda")).requires_grad_(False)

        # load textures_expr
        state_dict = checkpoint['t_expr']
        _state_dict = {
            k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state_dict.items()
        }
        try:
            self._textures_expr.load_state_dict(_state_dict, strict=False)
        except:
            print("[warning] t_expr weights are not loaded.")
        self._features_dc = [nn.Parameter(torch.tensor(_, dtype=torch.float, device="cuda")).requires_grad_(False) for _ in checkpoint['f_dc']]
        self._features_rest = [nn.Parameter(torch.tensor(_, dtype=torch.float, device="cuda")).requires_grad_(False) for _ in checkpoint['f_rest']]
        self._opacity = [nn.Parameter(torch.tensor(_, dtype=torch.float, device="cuda")).requires_grad_(False) for _ in checkpoint['opacity']]
        self._scaling = [nn.Parameter(torch.tensor(_, dtype=torch.float, device="cuda")).requires_grad_(False) for _ in checkpoint['scaling']]
        self._rotation = [nn.Parameter(torch.tensor(_, dtype=torch.float, device="cuda")).requires_grad_(False) for _ in checkpoint['rotation']]
        self.face_num = [torch.tensor(_, dtype=torch.long, device="cuda") for _ in checkpoint['face_num']]
        self.init_face_counter = torch.tensor(checkpoint['init_face_counter'], dtype=torch.int32, device="cuda")

        self.bc_gradient_accum = torch.zeros((self.bc.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = [torch.zeros((self.bc.shape[0], 1), device="cuda"), torch.zeros((self._xyz.shape[0], 1), device="cuda")]

        self.active_sh_degree = self.max_sh_degree

        self.split_faces = self.flame_model.split_faces

        print(f"Checkpoint loaded from {path}")

        if not kwargs['has_target']:
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

            if not self.disable_flame_static_offset:
                static_offset = self.flame_param['static_offset']
                static_offset = F.pad(static_offset, (0, 0, 0, 5023 - static_offset.shape[1]))
                new_static_offset = static_offset.squeeze(0)[self.flame_model.subdivide_edges]
                new_static_offset = new_static_offset.mean(dim=1).unsqueeze(0)
                static_offset = torch.cat((static_offset, new_static_offset), dim=1)
                num_verts = self.flame_model.v_template.shape[0]
                if static_offset.shape[0] != num_verts:
                    static_offset = F.pad(static_offset, (0, 0, 0, num_verts - static_offset.shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])
            self.flame_param['static_offset'] = static_offset
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param['translation'] = flame_param['translation']
            self.flame_param['rotation'] = flame_param['rotation']
            self.flame_param['neck_pose'] = flame_param['neck_pose']
            self.flame_param['jaw_pose'] = flame_param['jaw_pose']
            self.flame_param['eyes_pose'] = flame_param['eyes_pose']
            self.flame_param['expr'] = flame_param['expr']
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            raise NotImplementedError
            mask = (self.face_num[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.face_num = self.face_num[mask]
            self._xyz = self._xyz[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
    
    def load_checkpoint(self, path):    # TODO：实现
        raise NotImplementedError
    
    def save_checkpoint(self, path):
        raise NotImplementedError

    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def load_all_meshes(self, pose_meshes, num_frames=-1, fix_expr_id=-1):
        assert (num_frames == -1 and fix_expr_id == -1) or (num_frames > 0 and fix_expr_id >= 0)

        self.num_timesteps = max(pose_meshes) + 1 if num_frames == -1 else num_frames  # required by viewers
        num_verts = self.flame_model.v_template.shape[0]

        T = self.num_timesteps

        self.flame_param = {
            "shape": self.flame_param["shape"],
            "expr": torch.zeros([T, pose_meshes[0]["expr"].shape[1]]),
            "rotation": torch.zeros([T, 3]),
            "neck_pose": torch.zeros([T, 3]),
            "jaw_pose": torch.zeros([T, 3]),
            "eyes_pose": torch.zeros([T, 6]),
            "translation": torch.zeros([T, 3]),
            "static_offset": self.flame_param["static_offset"],
            "dynamic_offset": torch.zeros([T, num_verts, 3]),
        }

        fix_mesh = None if fix_expr_id == -1 else pose_meshes[fix_expr_id]
        for i in range(T):
            mesh = pose_meshes[i] if fix_mesh is None else fix_mesh
            self.flame_param["expr"][i] = torch.from_numpy(mesh["expr"])
            self.flame_param["rotation"][i] = torch.from_numpy(mesh["rotation"])
            self.flame_param["neck_pose"][i] = torch.from_numpy(mesh["neck_pose"])
            self.flame_param["jaw_pose"][i] = torch.from_numpy(mesh["jaw_pose"])
            self.flame_param["eyes_pose"][i] = torch.from_numpy(mesh["eyes_pose"])
            self.flame_param["translation"][i] = torch.from_numpy(mesh["translation"])
            # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])

        for k, v in self.flame_param.items():
            self.flame_param[k] = v.float().cuda()

        self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}