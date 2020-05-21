from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, MaxPool3D, \
    Conv3DTranspose, SeparableConv2D, MaxPool2D, Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.models import Model

def build_unet(img_shape, kernel_size, gf=64, channels=1):
    def conv_layer(x, filters, dropout=0, concat_layer=None):

        if concat_layer is not None:
            x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2,2), padding='same')(x)
            x = LeakyReLU()(x)
#             x = UpSampling2D(size=2)(x)
#             print(x.shape, concat_layer.shape)
            x = Concatenate()([x, concat_layer])

        ## This is the construction of conventional Unet
        c = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
        # Use dropout as in this report http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf and pix2pix paper
        if dropout > 0 and concat_layer is not None:
            c = Dropout(dropout)(c)
        c = Activation('relu')(c)
#         c = LeakyReLU()(c)
        c = BatchNormalization()(c)
        c = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(c)
        # Use dropout as in this report http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf and pix2pix paper
        if dropout > 0 and concat_layer is not None:
            c = Dropout(dropout)(c)
#         c = LeakyReLU()(c)
        c = Activation('relu')(c)
        # c = Activation('relu')(c)
        c = BatchNormalization()(c)
        # if dropout > 0:
        #     c = Dropout(dropout)(c)

        return c

    # Free-artifact image input

    arti = Input(shape=img_shape)

    down1 = conv_layer(arti, gf, 0)
    down1_mp = MaxPool2D(pool_size=(2, 2))(down1)
    down2 = conv_layer(down1_mp, 2 * gf)
    down2_mp = MaxPool2D(pool_size=(2, 2))(down2)
    down3 = conv_layer(down2_mp, 4 * gf)
    down3_mp = MaxPool2D(pool_size=(2, 2))(down3)
    down4 = conv_layer(down3_mp, 8 * gf)
    down4_mp = MaxPool2D(pool_size=(2, 2))(down4)
    down5 = conv_layer(down4_mp, 16 * gf)

    up4 = conv_layer(down5, 8 * gf, concat_layer=down4)
    up3 = conv_layer(up4, 4 * gf, concat_layer=down3)
    up2 = conv_layer(up3, 2 * gf, concat_layer=down2)
    up1 = conv_layer(up2, gf, concat_layer=down1)
    
#     gen_af = Conv2D(round(gf), kernel_size=9, strides=1, padding='same', activation='relu')(up1)
#     gen_af = BatchNormalization()(gen_af)
#     gen_af = Conv2D(round(gf/2), kernel_size=9, strides=1, padding='same', activation='relu')(up1)
#     gen_af = BatchNormalization()(gen_af)
    gen_af = Conv2D(1, kernel_size=1, strides=1, padding='same')(up1)
#     gen_af = LeakyReLU()(gen_af)
#     gen_af = BatchNormalization(momentum=0.8)(c)

    return Model(arti, gen_af)