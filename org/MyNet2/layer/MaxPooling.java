package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for max pooling layer.
 */
public class MaxPooling extends Pooling {
    /**
     * Constructor fot this challs.
     * @param channelNum Number of channel.
     * @param inShape Shape of input.
     * @param poolSize Size of pooling matrix.
     */
    public MaxPooling(int channelNum, int[] inShape, int poolSize){
        if (inShape.length != 2){
            this.exit("inShape length is wrong.");
        }
        this.channelNum = channelNum;
        this.inRow = inShape[0];
        this.inCol = inShape[1];
        this.outRow = inShape[0] / poolSize;
        this.outCol = inShape[1] / poolSize;
        this.poolSize = poolSize;
        this.name = "MaxPooling";
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix forward(Matrix in){
        int itr;
        double num, max;
        int kInMult = this.inRow * this.inCol;
        int kOutMult = this.outRow * this.outCol;
        int poolSizeI, poolSizej;

        Matrix rtn = new Matrix(in.row, this.channelNum * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.channelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        max = -100.0;
                        poolSizeI = this.poolSize * i;
                        poolSizej = this.poolSize * j;
                        for (int p = 0; p < this.poolSize; p++){
                            for (int q = 0; q < this.poolSize; q++){
                                itr = k*kInMult + (poolSizeI+p)*this.inCol + (poolSizej+q);
                                num = in.matrix[b][itr];

                                if (num > max){
                                    max = num;
                                }
                            }
                        }

                        rtn.matrix[b][k*kOutMult + i*this.outCol + j] = max;
                    }
                }
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "MaxPooling\n"
            + "channels: %d, pool size: %d", this.channelNum, this.poolSize
        );

        return str;
    }
}