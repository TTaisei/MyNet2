package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for pooling layer.
 */
public class Pooling extends Layer {
    /** Row of output. */
    public int outRow;
    /** Column of output. */
    public int outCol;
    /** Pooling matrix size. */
    public int poolSize;
    /** Number of channel. */
    public int channelNum;

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