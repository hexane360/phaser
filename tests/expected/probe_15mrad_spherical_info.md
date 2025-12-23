Generated from Kirkland's temsim code, with the following settings:

- Voltage: 200 kV
- 1024x1024 pixels
- Supercell size: 50x50 angstrom
- Convergence angle: 15 mrad

Aberrations:
```
Haider  Krivanek value
C3      C3,0     1 mm
C1      C1,0     -578.266 angstrom (Scherzer underfocus)
A1      C1,2     20.0+20.0j angstrom
3*S3    C3,2     0.15+0.2j mm
```

This corresponds to the below input to the 'probe' program:
```
0
probe_15mrad_spherical.tiff
1024
1024
50.0
50.0
200.0
1.0
0.0
578.266
15.0
0
0.0
0.0
C12a
0.0000020000
C12b
0.0000020000
C32a
0.1500000000
C32b
0.2000000000
END
```

Then, the following post-processing can be used to output the 'mag' (really amplitude) and 'phase' images:

```python
from pathlib import Path
import numpy
import tifffile

in_path = Path("probe_15mrad_spherical.tiff")
# read Kirkland FloatTIFF
img = numpy.asarray(tifffile.imread(f, series=1))
# combine real and imaginary images
re, im = numpy.split(img, 2, axis=-1)
img = re + im * 1.j
# center probe
img = numpy.fft.fftshift(img)
# normalize
img /= numpy.sqrt(numpy.sum(numpy.abs(img)**2))

# write output files
tifffile.imwrite(in_path.with_stem(in_path.stem + "_mag"), numpy.abs(img))
tifffile.imwrite(in_path.with_stem(in_path.stem + "_phase"), numpy.angle(img))
```