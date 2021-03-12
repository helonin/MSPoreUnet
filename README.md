# MSPoreU-net: A novel U-net architecture to exploit multi-scale pores in segmentation of digital rock images 
MSPoreU-net is a unet workflow for better results in performance, run time, the number of trainable parameters, and network predictions in rock image segmentation. To assess the performance of the proposed MSPoreU-net architecture, we have tested and evaluated it on three different sets of rock images including Tight-carbonate, Sandstone, and Carbonate. 

Schematic view of MSPoreU-net architecture in below figure. In this model, the sequences of three convolutional layers in the U-net architecture are replaced with the MSPore blocks. Furthermore, instead of using plain skip connections, the proposed MSPoreSkip block sequence are used.

![MSPoreModel](https://user-images.githubusercontent.com/50166193/110905845-631d7880-8320-11eb-97f3-61ef3e40d61a.jpg)

The required packages to use this python repository are: 'os','numpy', 'scipy', 'h5py', 'tensorflow', 'matplotlib', 'keras', 'skimage', 'cv2', and 'pandas'. I recommend to use Anaconda which has all these packages installed except cv2 and tensorflow of which you can easily install from pip.

Example:
Comparation of different pore scales: (a). Tight-carbonate sample has fine-size pores, (b). Sandstone sample has medium-size pores and (c). Carbonate sample has multi-scale coarse-size pores.

![Imagesforcompare](https://user-images.githubusercontent.com/50166193/110902698-872a8b00-831b-11eb-8b3b-8aba151875da.jpg)


Mohsen Abdolahzadeh Kondori
University of Tehran
Phone:+989150465172
Email: MohsenKondori@ut.ac.ir
       MohsenKondori@yahoo.com
