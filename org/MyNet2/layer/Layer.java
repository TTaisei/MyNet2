package org.MyNet2.layer;

import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for layer.
 */
public class Layer {
    /** Type of activation function for this layer. */
    public AFType afType;
    /** Activation function of this layer. */
    public ActivationFunction actFunc;
    /** Name of this layer's activation function. */
    public String actFuncName;
    /** Name of this layer */
    public String name = null;

    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public Layer(){
        ;
    }


    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    public Matrix forward(Matrix in){
        return in.clone();
    }

    /**
     * Doing back propagation.
     */
    public void back(){
        ;
    }

    /**
     * Calucrate delta each nodes.
     */
    public void calDelta(){
        ;
    }

    @Override
    public String toString(){
        return "";
    }
}