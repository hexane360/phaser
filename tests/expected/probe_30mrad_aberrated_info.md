Generated from Kirkland's temsim code, with the following settings:

- Voltage: 300 kV
- 1024x1024 pixels
- Supercell size: 50x50 angstrom
- Convergence angle: 30 mrad

Aberrations:
```
Haider  Krivanek  value
A1      C1,2      10+10j angstrom
3*B2    C2,1      1e3+2e3j angstrom
3*S3    C3,2      50e3j angstrom
```

This corresponds to the below input to the 'probe' program:
```
0
probe_30mrad_aberrated.tiff
1024
1024
50.0
50.0
300.0
0.0
0.0
0.0
30.0
0
0.0
0.0
C12a
0.0000010000
C12b
0.0000010000
C21a
0.0001000000
C21b
0.0002000000
C32a
0.0000000000
C32b
0.0050000000
END
```

Then, the following post-processing can be used to output the 'mag' (really amplitude) and 'phase' images:

```python
from pathlib import Path
import numpy
import tifffile

in_path = Path("probe_30mrad_aberrated.tiff")
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