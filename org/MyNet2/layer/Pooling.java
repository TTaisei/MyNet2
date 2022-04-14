package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for pooling layer.
 */
public class Pooling extends Layer {
    /** Row of input. */
    public int inRow;
    /** Column of input. */
    public int inCol;
    /** Pooling matrix size. */
    public int poolSize;

    public Pooling(){
        ;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix forward(Matrix in){
        return in.clone();
    }

    /**
     * Doing back propagation.
     */
    @Override
    public void back(){
        ;
    }
}