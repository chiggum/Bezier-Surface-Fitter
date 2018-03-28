# Bezier-Surface-Fitter

Fits Bezier surface over each channel of image where the height at `(x,y)` for channel `c` is the pixel value of the image at `(x,y)` for channel `c`.

- The motivation is to represent a digital image as a continuous (analog) image.
- Uses Keras with Tensorflow backend to fit Bezier surface of degree `(m,n)` provided by user.
- The learned control points (`(m+1)*(n+1)` in total) can act as a low-dimensional representation of the image (if `(m+1)*(n+1)<H*W`).
