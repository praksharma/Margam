# ParaView playback test dataset

Synthetic known-value field, not a PINN prediction. Open standing-wave.pvd in
ParaView, Apply, colour by demo_scalar and Play. Keep the PVD and companion
folder together. Contains 21 times on a 21x21 regular spatial grid.

Formula: sin(pi*x)*sin(pi*y)*cos(2*pi*t), spatial domain [-1,1]^2, t in [0,1].
Time is dimensionless for this artificial demo. Real exports use supplied physical
times, never optimiser iterations. validation.json records real VTK 9.7.1 reader
checks for geometry, field values and timestamps. ParaView GUI playback has not
been exercised on this host. Regenerate with examples.export_paraview_demo.
