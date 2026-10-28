"""Small scalar PINN expression language. AST interpretation; no Python eval/imports."""
import ast
import math


class Expression:
    def __init__(self, text, dimensions, *, target=False, fields=('u',), coordinates=None):
        if not isinstance(text,str) or not 0 < len(text.strip()) <= 512:
            raise ValueError('Expressions need 1–512 characters.')
        self.dimensions=dimensions;self.target=target;self.fields=tuple(fields)
        self.coordinates=tuple(coordinates or ('x','y')[:dimensions])
        if self.coordinates not in (('x',),('x','y'),('x','t')) or len(self.coordinates)!=dimensions:raise ValueError('Unsupported coordinate contract.')
        if self.fields not in (('u',),('u','v','p')):raise ValueError('Unsupported field contract.')
        try:self.root=ast.parse(text,mode='eval').body
        except SyntaxError as error:raise ValueError('Invalid mathematical expression.') from error
        if sum(1 for _ in ast.walk(self.root))>128:raise ValueError('Expression is too complex.')
        self.check(self.root)
        self.uses_normal=any(isinstance(node,ast.Name) and node.id in ('nx','ny','dn') for node in ast.walk(self.root))

    def check(self,node,depth=0):
        if depth>20:raise ValueError('Expression nesting is too deep.')
        if isinstance(node,ast.Constant):
            if type(node.value) not in (int,float) or not math.isfinite(node.value) or abs(node.value)>1e6:raise ValueError('Use finite numeric constants up to 1e6.')
        elif isinstance(node,ast.Name):
            names={'c','nx','ny','pi','nu'} | set(self.coordinates) | (set() if self.target else set(self.fields))
            if node.id not in names:raise ValueError('Unknown expression name: '+node.id)
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            self.check(node.operand,depth+1)
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            if isinstance(node.op,ast.Pow):
                exponent=node.right
                if isinstance(exponent,ast.UnaryOp) and isinstance(exponent.op,(ast.UAdd,ast.USub)):exponent=exponent.operand
                if not isinstance(exponent,ast.Constant) or type(exponent.value) not in (int,float) or abs(exponent.value)>8:raise ValueError('Powers need a constant exponent between -8 and 8.')
            self.check(node.left,depth+1);self.check(node.right,depth+1)
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            name=node.func.id
            functions={'sin','cos','exp','tanh','sqrt','abs'} | (set() if self.target else {'D','laplace','dn'})
            if name not in functions or len(node.args)!=(2 if name=='D' else 1):raise ValueError('Unsupported function or argument count: '+name)
            if name=='D' and (not isinstance(node.args[1],ast.Name) or node.args[1].id not in self.coordinates):raise ValueError('D needs a coordinate name as its second argument.')
            for arg in node.args:self.check(arg,depth+1)
        else:raise ValueError('Only arithmetic, named coordinates and supported mathematical functions are allowed.')

    def __call__(self, points, u=None, *, source=0., normal=None, viscosity=1.):
        import torch
        def derivative(value,index):
            if not isinstance(value,torch.Tensor) or not value.requires_grad:return points.new_zeros((len(points),1))
            grad=torch.autograd.grad(value.sum(),points,create_graph=True,retain_graph=True,allow_unused=True)[0]
            return points.new_zeros((len(points),1)) if grad is None else grad[:,index:index+1]
        normal=normal or (0.,)*self.dimensions
        values={'x':points[:,0:1],'u':u,'c':source,'nx':normal[0],'ny':normal[1] if len(normal)>1 else 0.,'pi':math.pi,'nu':viscosity}
        if u is not None:
            for index,field in enumerate(self.fields):values[field]=u[:,index:index+1]
        values.update({name:points[:,index:index+1] for index,name in enumerate(self.coordinates)})
        def visit(node):
            if isinstance(node,ast.Constant):return node.value
            if isinstance(node,ast.Name):return values[node.id]
            if isinstance(node,ast.UnaryOp):return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
            if isinstance(node,ast.BinOp):
                a,b=visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Add):return a+b
                if isinstance(node.op,ast.Sub):return a-b
                if isinstance(node.op,ast.Mult):return a*b
                if isinstance(node.op,ast.Div):return a/b
                return a**b
            name=node.func.id;value=visit(node.args[0])
            if name=='D':return derivative(value,self.coordinates.index(node.args[1].id))
            if name=='dn':return sum(normal[i]*derivative(value,i) for i in range(self.dimensions))
            if name=='laplace':return sum(derivative(derivative(value,i),i) for i,name in enumerate(self.coordinates) if name!='t')
            tensor=torch.as_tensor(value,dtype=points.dtype,device=points.device)
            return getattr(torch,name)(tensor)
        try:result=visit(self.root)
        except ArithmeticError as error:raise ValueError('Expression evaluation failed: '+str(error)) from error
        if not isinstance(result,torch.Tensor):return points.new_full((len(points),1),float(result))
        return result.expand(len(points),1) if result.ndim==0 else result
