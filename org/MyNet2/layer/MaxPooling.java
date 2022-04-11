package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for max pooling layer.
 */
public class MaxPooling extends Pooling {
    /**
     * Constructor fot this challs.
     * @param channelNum Number of channel.
     * @param poolSize Size of pooling matrix.
     */
    public MaxPooling(int channelNum, int poolSize){
        this.channelNum = channelNum;
        this.poolSize = poolSize;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix4d forward(Matrix4d in){
        int batchSize = in.shape[0];
        int rtnRow = in.shape[2] / this.poolSize;
        int rtnCol = in.shape[3] / this.poolSize;
        double num, max;

        Matrix4d rtn = new Matrix4d(
            new int[]{
                batchSize,
                this.channelNum,
                rtnRow,
                rtnCol
            }
        );
        for (int b = 0; b < batchSize; b++){
            for (int k = 0; k < this.channelNum; k++){
                for (int i = 0; i < rtnRow; i++){
                    for (int j = 0; j < rtnCol; j++){
                        max = -100.0;
                        for (int p = 0; p < this.poolSize; p++){
                            for (int q = 0; q < this.poolSize; q++){
                                num = in.matrix.get(b).matrix.get(k).matrix[this.poolSize*i+p][this.poolSize*j+q];

                                if (num > max){
                                    max = num;
                                }
                            }
                        }

                        rtn.matrix.get(b).matrix.get(k).matrix[i][j] = max;
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