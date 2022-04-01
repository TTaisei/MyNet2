package org.MyNet2.AF;

import org.MyNet2.*;

/**
 * Activation function's base class.
 * All activation functions must extend this class.
 */
public class ActivationFunction {
    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public ActivationFunction(){
        ;
    }

    /**
     * Execute this actiation function.
     * @param in linear transformationed matrix.
     * @return output matrix.
     */
    public Matrix calc(Matrix in){
        Matrix rtn = in.clone();
        return rtn;
    }

    /**
     * Calcurate this activation function's differential.
     * @param in Matrix of input.
     * @return The result of differentiating this activation function.
     */
    public Matrix diff(Matrix in){
        Matrix rtn = in.clone();
        return rtn;
    }

    @Override
    public String toString(){
        return "";
    }
}