"""Deterministic diagram placement; never modifies connections or physics."""

def positions(nodes,wires,operation):
    nodes=sorted(nodes,key=lambda n:(n.pos().y(),n.pos().x(),n.identifier))
    if len(nodes)<2:return {}
    result={n:(n.pos().x(),n.pos().y()) for n in nodes}
    if operation=='auto':
        # Kahn layers; cyclic leftovers remain a visible final column, never recurse.
        pending=set(nodes);levels={};level=0
        parents={n:{w.source.parentItem() for w in wires if w.target.parentItem() is n and w.source.parentItem() in pending} for n in nodes}
        while pending:
            ready=[n for n in nodes if n in pending and not (parents[n]&pending)]
            if not ready:ready=[n for n in nodes if n in pending]
            for n in ready:levels[n]=level
            pending.difference_update(ready);level+=1
        x=min(n.pos().x() for n in nodes);y=min(n.pos().y() for n in nodes)
        columns={}
        for n in nodes:columns.setdefault(levels[n],[]).append(n)
        offset=x
        for depth in sorted(columns):
            column=columns[depth]
            row=y
            for n in column:
                result[n]=(offset,row);row+=n.rect().height()+50
            offset+=max(n.rect().width() for n in column)+100
        return result
    if operation in ('horizontal','vertical'):
        if len(nodes)<3:return {}
        axis=0 if operation=='horizontal' else 1
        size=lambda n:n.rect().width() if axis==0 else n.rect().height()
        ordered=sorted(nodes,key=lambda n:(result[n][axis],n.identifier))
        first,last=ordered[0],ordered[-1]
        extent=result[last][axis]+size(last)-result[first][axis]
        gap=(extent-sum(size(n) for n in ordered))/(len(ordered)-1)
        # Do not create overlap when the selected span is too small.
        if gap<0:return {}
        cursor=result[first][axis]
        for n in ordered:
            xy=list(result[n]);xy[axis]=cursor;result[n]=tuple(xy);cursor+=size(n)+gap
        return result
    axis=0 if operation in ('left','center','right') else 1
    size=lambda n:n.rect().width() if axis==0 else n.rect().height()
    low=min(result[n][axis] for n in nodes)
    high=max(result[n][axis]+size(n) for n in nodes)
    for n in nodes:
        xy=list(result[n])
        xy[axis]=low if operation in ('left','top') else high-size(n) if operation in ('right','bottom') else (low+high-size(n))/2
        result[n]=tuple(xy)
    return result
