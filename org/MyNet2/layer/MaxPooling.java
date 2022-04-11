package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for max pooling layer.
 */
public class MaxPooling extends Pooling {
    /**
     * Constructor for this class.
     * @param inShape Input matrix shape.
     * @param poolShape pooling matrix shape.
     */
    public MaxPooling(int[] inShape, int[] poolShape){
        this.setup(inShape, poolShape, 1);
    }

    /**
     * Constructor for this class.
     * @param inShape Input matrix shape.
     * @param poolShape pooling matrix shape.
     * @param stride Number of stride.
     */
    public MaxPooling(int[] inShape, int[] poolShape, int stride){
        this.setup(inShape, poolShape, stride);
    }

    /**
     * Construct instead of constructor.
     * @param inShape Input matrix shape.
     * @param poolShape pooling matrix shape.
     * @param stride Number of stride.
     */
    protected void setup(int[] inShape, int[] poolShape, int stride){
        if (inShape.length != 4){
            this.exit("inShape length is wrong.");
        }else if (poolShape.length != 2){
            this.exit("poolShape length is wrong.");
        }

        this.inShape = inShape;
        this.poolShape = poolShape;
        this.outShape = new int[]{
            this.inShape[0],
            this.inShape[1],
            this.inShape[2] / this.poolShape[0] / stride,
            this.inShape[3] / this.poolShape[1] / stride
        };
        this.stride = stride;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix4d forward(Matrix4d in){
        Matrix4d rtn = new Matrix4d(this.outShape);
        double num, max;

        for (int b = 0; b < in.shape[0]; b++){
            for (int k = 0; k < in.shape[1]; k++){
                for (int i = 0; i < rtn.shape[2]; i++){
                    for (int j = 0; j < rtn.shape[3]; j++){
                        max = -100.0;
                        for (int p = 0; p < this.poolShape[0]; p++){
                            for (int q = 0; q < this.poolShape[1]; q++){
                                num = in.matrix.get(b).matrix.get(k).matrix[this.poolShape[0]*i+p][this.poolShape[1]*j+q];

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
}