# Data

`caltech10fct.npz` - **Caltech-10** as used in the paper: the ten largest object classes of Caltech-101 (airplanes,
Motorbikes, Faces, watch, Leopards, bonsai, car_side, ketch, chandelier, hawksbill), every picture (2,744 training,
300 test = 30 per class), 4 % cut off every side, centre square, resized to 32 x 32 colour (box filter). Made by
`src/prepare_caltech101.py --all --crop --trim 0.04`.

Derived from **Caltech-101** (L. Fei-Fei, R. Fergus, P. Perona; CaltechDATA record mzrjq-6wc02,
https://doi.org/10.22002/D1.20086), licence **CC BY 4.0**. Changes: the class selection, trim, crop and resize above.
`demo/seeing_why.py` `prepare_caltech10()` downloads the original (MD5 in `caltech101/MD5SUMS`) and rebuilds the set
for comparison.
