## Crop eye pair from face image using HAAR Cascade and/or MTCNN
-------------

### How to run?
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. If input face images are in directory `example/input/dir` and output direcotry `example/output/dir` then run the following command which will preserve input directory structure for further tasks.

    ```
    python crop_eye_pair.py --input_dir 'example/input/dir' --output_dir 'example/output/dir' --method 'mtcnn' --device 'cuda'
    ```

4. Supported `method` options: `haar-cascade` and `mtcnn`
5. Supported `device` options: `cpu` and `cuda`. 

Notes: Setting device to `cuda` has no effect while using `haar-cascade` method.


## Credits

1. MTCNN code borrowed from https://github.com/khrlimam/mtcnn-pytorch
2. HAAR Cascade code inspired from https://github.com/zekeriyafince/EyePair