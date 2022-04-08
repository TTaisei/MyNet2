package org.MyNet2.layer;

import ort.MyNet2.*;

/**
 * Class for max pooling layer.
 */
public class MaxPooling extends Pooling {
    /**
     * Constructor for this class.
     * @param poolRow 
     * @param poolCol
     */
    public MaxPooling(int poolRow, int poolCol){
        this.poolRow = poolRow;
        this.poolCol = poolCol;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix4d forward(Matrix4d in){
        return in.clone();
    }
}