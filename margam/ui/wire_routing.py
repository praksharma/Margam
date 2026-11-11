"""Bounded orthogonal detours for the current left-input/right-output blocks.

This avoids rectangles, not other wires. It never edits diagram geometry.
"""
import math

MAX_BLOCKS=64
CLEARANCE=12.
STUB=24.


def crosses(a,b,rect):
    """Whether an axis-aligned segment crosses the open interior of a rectangle."""
    left,top,right,bottom=rect
    if a[0]==b[0]:return left<a[0]<right and max(min(a[1],b[1]),top)<min(max(a[1],b[1]),bottom)
    if a[1]==b[1]:return top<a[1]<bottom and max(min(a[0],b[0]),left)<min(max(a[0],b[0]),right)
    return True


def route(start,end,rectangles,source,target):
    """Return a clear polyline, or None when the bounded lane search cannot route.

    Rectangles are (left, top, right, bottom); source/target are their indices.
    Short horizontal port stubs stay fixed. Search the two elbow paths and the
    top/bottom lanes of every expanded block, then choose a shortest clear path.
    """
    if len(rectangles)>MAX_BLOCKS:return None
    if not (0<=source<len(rectangles) and 0<=target<len(rectangles)):return None
    if any(not math.isfinite(v) for point in (start,end,*rectangles) for v in point):return None
    obstacles=[(l-CLEARANCE,t-CLEARANCE,r+CLEARANCE,b+CLEARANCE) for l,t,r,b in rectangles]
    a=(start[0]+STUB,start[1]);b=(end[0]-STUB,end[1])
    candidates=[(start,a,(b[0],a[1]),b,end),(start,a,(a[0],b[1]),b,end)]
    lanes=sorted({y for rect in obstacles for y in (rect[1],rect[3])})
    candidates.extend((start,a,(a[0],y),(b[0],y),b,end) for y in lanes)
    clear=[]
    for points in candidates:
        if any(crosses(p,q,rect) for j,(p,q) in enumerate(zip(points,points[1:]))
               for i,rect in enumerate(obstacles)
               if not (i==source and j==0 or i==target and j==len(points)-2)):
            continue
        # Remove zero-length/collinear points only after checking port-stub escapes.
        result=[]
        for point in points:
            if result and point==result[-1]:continue
            while len(result)>1 and (result[-2][0]==result[-1][0]==point[0] or result[-2][1]==result[-1][1]==point[1]):
                # Do not collapse a reversal: it can re-enter a block.
                u,v=result[-2:]
                if (v[0]-u[0])*(point[0]-v[0])+(v[1]-u[1])*(point[1]-v[1])<0:break
                result.pop()
            result.append(point)
        length=sum(abs(p[0]-q[0])+abs(p[1]-q[1]) for p,q in zip(result,result[1:]))
        clear.append((length,len(result),tuple(result)))
    return list(min(clear)[2]) if clear else None
