package org.myNet2.actFunc;

import java.io.Serializable;
import org.myNet2.*;

/**
 * Activation function's base class.
 * All activation functions must extend this class.
 */
public class ActivationFunction implements Serializable {
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
