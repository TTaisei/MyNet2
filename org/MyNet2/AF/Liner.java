package org.MyNet.af;

import org.MyNet2.*;

/**
 * Liner function.
 */
public class Liner extends ActivationFunction {
    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public Liner(){
        ;
    }

    /**
     * Execute this actiation function.
     * @param in linear transformationed matrix.
     * @return output matrix.
     */
    @Override
    public Matrix calc(Matrix in){
        Matrix rtn = in.clone();
        return rtn;
    }

    /**
     * Calcurate this activation function's differential.
     * @param in Matrix of input.
     * @return The result of differentiating this activation function.
     */
    @Override
    public Matrix diff(Matrix in){
        Matrix rtn = new Matrix(in.row, in.col, 1.0);
        return rtn;
    }

    @Override
    public String toString(){
        return "Liner";
    }
}