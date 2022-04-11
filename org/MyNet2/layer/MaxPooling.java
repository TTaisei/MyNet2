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
        this.outRow = inShape[0] / poolSize;
        this.outCol = inShape[1] / poolSize;
        this.poolSize = poolSize;
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

        Matrix rtn = new Matrix(in.row, this.channelNum * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.channelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        max = -100.0;
                        for (int p = 0; p < this.poolSize; p++){
                            for (int q = 0; q < this.poolSize; q++){
                                itr = k*this.channelNum + (this.poolSize*i+p)*this.outRow + (this.poolSize*j+q);
                                num = in.matrix[b][itr];

                                if (num > max){
                                    max = num;
                                }
                            }
                        }

                        rtn.matrix[b][k*this.channelNum + i*this.outRow + j*this.outCol] = max;
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