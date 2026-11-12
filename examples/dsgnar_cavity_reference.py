"""Independent Re=100 smooth-lid reference: second-order vorticity/streamfunction.

No upstream data is read. This grid solution has discretization error and is not
Dedalus or an exact reference. Generate two resolutions to quantify that error.
Arrays are indexed [y,x]; psi_y=u, -psi_x=v, omega=-laplace(psi).
"""
import argparse
from pathlib import Path
import time
import numpy as np


def poisson(omega,h):
    from scipy.fft import dstn,idstn
    n=omega.shape[0]-2
    k=np.arange(1,n+1)
    eigen=(2-2*np.cos(np.pi*k/(n+1)))/h**2
    psi=np.zeros_like(omega)
    psi[1:-1,1:-1]=idstn(dstn(omega[1:-1,1:-1],type=1)/(eigen[:,None]+eigen[None,:]),type=1)
    return psi


def fields(omega,h):
    psi=poisson(omega,h);u=np.zeros_like(psi);v=np.zeros_like(psi)
    u[1:-1,1:-1]=(psi[2:,1:-1]-psi[:-2,1:-1])/(2*h)
    v[1:-1,1:-1]=-(psi[1:-1,2:]-psi[1:-1,:-2])/(2*h)
    x=np.linspace(0,1,len(psi));u[-1,:]=4*x*(1-x)
    return psi,u,v


def rhs(omega,h,viscosity):
    psi,u,v=fields(omega,h)
    # Thom wall closure: zero normal flow, prescribed tangential velocity.
    w=omega.copy();w[0,1:-1]=-2*psi[1,1:-1]/h**2
    w[-1,1:-1]=-2*psi[-2,1:-1]/h**2-2*u[-1,1:-1]/h
    w[1:-1,0]=-2*psi[1:-1,1]/h**2;w[1:-1,-1]=-2*psi[1:-1,-2]/h**2
    core=w[1:-1,1:-1]
    lap=(w[2:,1:-1]+w[:-2,1:-1]+w[1:-1,2:]+w[1:-1,:-2]-4*core)/h**2
    advection=u[1:-1,1:-1]*(w[1:-1,2:]-w[1:-1,:-2])/(2*h)+v[1:-1,1:-1]*(w[2:,1:-1]-w[:-2,1:-1])/(2*h)
    return viscosity*lap-advection


def solve(grid=129,tolerance=1e-8,max_time=100.):
    if type(grid) is not int or not 17<=grid<=513:raise ValueError('Grid must have 17–513 nodes per axis.')
    if not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(max_time) or max_time<=0:raise ValueError('Positive finite tolerance and maximum physical time required.')
    h=1/(grid-1);nu=.01;dt=min(.18*h*h/nu,.2*h)
    w=np.zeros((grid,grid));started=time.perf_counter();converged=False
    for step in range(1,int(np.ceil(max_time/dt))+1):
        k1=rhs(w,h,nu);trial=w.copy();trial[1:-1,1:-1]+=dt*k1
        k2=rhs(trial,h,nu);w[1:-1,1:-1]+=.5*dt*(k1+k2)
        if not np.isfinite(w).all():raise FloatingPointError('Reference integration diverged.')
        if step%250==0:
            residual=float(np.max(np.abs(rhs(w,h,nu))))
            if residual<tolerance:converged=True;break
    psi,u,v=fields(w,h);residual=float(np.max(np.abs(rhs(w,h,nu))))
    return dict(x=np.linspace(0,1,grid),y=np.linspace(0,1,grid),u=u,v=v,psi=psi,
                steady_residual=residual,converged=converged,steps=step,physical_time=step*dt,seconds=time.perf_counter()-started,
                method='Independent second-order centered streamfunction/vorticity; Thom walls; Heun pseudo-time',reynolds=100.)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--grid',type=int,default=129)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tolerance',type=float,default=1e-8)
    args=parser.parse_args();result=solve(args.grid,args.tolerance)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**result)
    print({k:v for k,v in result.items() if not isinstance(v,np.ndarray)},flush=True)
    if not result['converged']:raise RuntimeError('Reference did not reach steady tolerance.')


if __name__=='__main__':main()
