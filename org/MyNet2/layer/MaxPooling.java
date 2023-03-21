package org.myNet2.layer;

import org.myNet2.*;

/**
 * Class for max pooling layer.
 */
public class MaxPooling extends Pooling {
    /**
     * Constructor for this class.
     * @param poolSize Size of pooling matrix.
     */
    public MaxPooling(int poolSize){
        this.name = "MaxPooling";
        this.poolSize = poolSize;
    }

    /**
     * Constructor fot this class.
     * @param channelNum Number of channel.
     * @param inShape Shape of input.
     * @param poolSize Size of pooling matrix.
     */
    public MaxPooling(int channelNum, int[] inShape, int poolSize){
        this.setup(channelNum, inShape, poolSize);
    }

    /**
     * Construct instead of constructor.
     * @param channelNum Number of channel.
     * @param inShape Shape of input.
     * @param poolSize Size of pooling matrix.
     */
    public void setup(int channelNum, int[] inShape, int poolSize){
        if (inShape.length != 2){
            this.exit("inShape length is wrong.");
        }
        this.name = "MaxPooling";
        this.channelNum = channelNum;
        this.inRow = inShape[0];
        this.inCol = inShape[1];
        this.outRow = inShape[0] / poolSize;
        this.outCol = inShape[1] / poolSize;
        this.poolSize = poolSize;
        this.delta = new Matrix(this.channelNum, this.inRow*this.inCol);
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
        int poolSizeI, poolSizeJ;

        this.x = new Matrix(in.row, this.channelNum * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.channelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        max = -100.0;
                        poolSizeI = this.poolSize * i;
                        poolSizeJ = this.poolSize * j;
                        for (int p = 0; p < this.poolSize; p++){
                            for (int q = 0; q < this.poolSize; q++){
                                itr = k*kInMult + (poolSizeI+p)*this.inCol + (poolSizeJ+q);
                                num = in.matrix[b][itr];

                                if (num > max){
                                    max = num;
                                }
                            }
                        }

                        this.x.matrix[b][k*kOutMult + i*this.outCol + j] = max;
                    }
                }
            }
        }

        this.a = this.x;

        return this.a.clone();
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "MaxPooling\nact: null\n"
            + "%d, %d, %d => (%d, %d) => %d, %d, %d",
            this.channelNum, this.inRow, this.inCol, this.poolSize, this.poolSize, this.channelNum, this.outRow, this.outCol
        );

        return str;
    }
}
