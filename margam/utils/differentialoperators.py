"""Autograd differential operators preserving solution dtype, device and higher-order graphs."""
import torch

__all__ = (
    'convective',
    'div',
    'grad',
    'jac',
    'laplacian',
    'matrix_div',
    'normal_derivative',
    'partial',
    'rot',
    'sym_grad',
)


def _grad_or_zeros(output_sum, variable, prototype):
    if not output_sum.requires_grad:
        return None
    return torch.autograd.grad(
        output_sum,
        variable,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]


def laplacian(model_out, *derivative_variable, grad=None):
    """Computes the Laplacian of a scalar output with respect to the given variable(s)."""
    out = model_out.new_zeros((*model_out.shape[:-1], 1))
    for vari in derivative_variable:
        g = grad
        if g is None or len(derivative_variable) > 1:
            g = _grad_or_zeros(model_out.sum(), vari, model_out)
            if g is None:
                continue
        if not g.requires_grad:
            continue
        for i in range(vari.shape[-1]):
            comp = g.narrow(-1, i, 1)
            d2u = _grad_or_zeros(comp.sum(), vari, model_out)
            if d2u is not None:
                out = out + d2u.narrow(-1, i, 1)
    return out


def grad(model_out, *derivative_variable):
    """Computes the gradient of a scalar output with respect to the given variable(s)."""
    grads = []
    for vari in derivative_variable:
        new_grad = _grad_or_zeros(model_out.sum(), vari, model_out)
        grads.append(model_out.new_zeros(vari.shape) if new_grad is None else new_grad)
    return torch.column_stack(grads)


def normal_derivative(model_out, normals, *derivative_variable):
    """Computes the directional/normal derivative with respect to given normal vectors."""
    gradient = grad(model_out, *derivative_variable)
    normal_tensor = (
        normals.to(device=gradient.device, dtype=gradient.dtype)
        if isinstance(normals, torch.Tensor)
        else gradient.new_tensor(normals)
    )
    return (gradient * normal_tensor).sum(dim=-1, keepdim=True)


def div(model_out, *derivative_variable):
    """Computes the divergence of a vector-valued output with respect to the given variable(s)."""
    divergence = model_out.new_zeros((derivative_variable[0].shape[0], 1))
    var_dim = 0
    for vari in derivative_variable:
        width = vari.shape[1]
        for i in range(width):
            comp = model_out.narrow(1, var_dim + i, 1)
            du = _grad_or_zeros(comp.sum(), vari, model_out)
            if du is not None:
                divergence = divergence + du.narrow(1, i, 1)
        var_dim += width
    return divergence


def jac(model_out, *derivative_variable):
    """Computes the Jacobian (b, m, n) of an output with respect to the given input(s)."""
    du_rows = []
    for i in range(model_out.shape[1]):
        comp = model_out[:, i]
        du_i = []
        for vari in derivative_variable:
            d = _grad_or_zeros(comp.sum(), vari, model_out)
            du_i.append(model_out.new_zeros(vari.shape) if d is None else d)
        du_rows.append(torch.cat(du_i, dim=1))
    return torch.stack(du_rows, dim=1)


def rot(model_out, *derivative_variable):
    """Computes the 3D rotation/curl of a 3-component vector field."""
    jacobian = jac(model_out, *derivative_variable)
    return torch.stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        ),
        dim=1,
    )


def partial(model_out, *derivative_variables):
    """Computes the (n-th, possibly mixed) partial derivative with respect to given variable(s)."""
    du = model_out
    for inp in derivative_variables:
        d = _grad_or_zeros(du.sum(), inp, model_out)
        if d is None:
            return model_out.new_zeros(inp.shape)
        du = d
    return du


def convective(deriv_out, convective_field, *derivative_variable):
    r"""Computes the convective term :math:`(v \cdot \nabla)u`."""
    jac_x = jac(deriv_out, *derivative_variable)
    velocity = convective_field.to(device=jac_x.device, dtype=jac_x.dtype)
    return torch.bmm(jac_x, velocity.unsqueeze(dim=2)).squeeze(dim=2)


def sym_grad(model_out, *derivative_variable):
    r"""Computes the symmetric gradient: :math:`0.5(\nabla u + \nabla u^T)`."""
    jac_matrix = jac(model_out, *derivative_variable)
    return 0.5 * (jac_matrix + torch.transpose(jac_matrix, 1, 2))


def matrix_div(model_out, *derivative_variable):
    """Computes the row-wise divergence for matrix/tensor-valued functions."""
    if model_out.shape[1] == 0:
        return model_out.new_zeros((len(model_out), 0))
    rows = [
        div(model_out.narrow(1, i, 1).squeeze(1), *derivative_variable)
        for i in range(model_out.shape[1])
    ]
    return torch.cat(rows, dim=1)
