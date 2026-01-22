from .render import render
from .tex_render import tex_render
from .hybrid_render import hybrid_render

type2render_func = dict(
    render = render,
    tex_render = tex_render,
    hybrid_render = hybrid_render
)

def create_render_func(type):
    return type2render_func[type]