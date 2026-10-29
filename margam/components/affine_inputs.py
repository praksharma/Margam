"""Parse bounded affine coordinate expressions without evaluating Python code."""
import ast,math


def affine_map(text,coordinates):
    if not isinstance(text,str) or len(text)>1024:raise ValueError('Affine input mapping must be text up to 1024 characters.')
    expressions=[part.strip() for part in text.split(';')] if text.strip() else list(coordinates)
    if not 1<=len(expressions)<=16 or any(not part for part in expressions):raise ValueError('Use 1–16 affine expressions separated by semicolons.')
    dimension=len(coordinates)
    def parse(node):
        if isinstance(node,ast.Name):
            if node.id=='pi':return [0.]*dimension+[math.pi]
            if node.id not in coordinates:raise ValueError('Unknown affine coordinate: '+node.id)
            return [float(name==node.id) for name in coordinates]+[0.]
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return [0.]*dimension+[float(node.value)]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=parse(node.operand);return [(-v if isinstance(node.op,ast.USub) else v) for v in value]
        if isinstance(node,ast.BinOp):
            a,b=parse(node.left),parse(node.right)
            if isinstance(node.op,(ast.Add,ast.Sub)):return [x+(-y if isinstance(node.op,ast.Sub) else y) for x,y in zip(a,b)]
            if isinstance(node.op,ast.Mult) and (not any(a[:-1]) or not any(b[:-1])):
                scalar,value=(a[-1],b) if not any(a[:-1]) else (b[-1],a)
                return [scalar*v for v in value]
            if isinstance(node.op,ast.Div) and not any(b[:-1]) and b[-1]!=0:return [v/b[-1] for v in a]
        raise ValueError('Use affine coordinate arithmetic only: constants, pi, +, −, multiplication/division by constants.')
    rows=[]
    for expression in expressions:
        try:
            tree=ast.parse(expression,mode='eval')
            if sum(1 for _ in ast.walk(tree))>128:raise ValueError('Affine expression is too complex.')
            row=parse(tree.body)
        except (SyntaxError,RecursionError,OverflowError,ZeroDivisionError) as error:raise ValueError('Invalid affine expression.') from error
        if any(not math.isfinite(v) or abs(v)>1e6 for v in row):raise ValueError('Affine coefficients must be finite and at most 1e6 in magnitude.')
        rows.append(row)
    return [row[:-1] for row in rows],[row[-1] for row in rows]
