package org.MyNet2.layer;

import java.io.Serializable;
import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for layer.
 */
public class Layer implements Serializable {
    /** The list of weight for this layer */
    public Matrix w;
    /** Type of activation function for this layer. */
    public AFType afType;
    /** Activation function of this layer. */
    public ActivationFunction actFunc;
    /** Name of this layer's activation function. */
    public String actFuncName;
    /** Delta. */
    public Matrix delta;
    /** Name of this layer */
    public String name = null;

    /** Linear transformed matrix. */
    public Matrix x;
    /** Matrix of output from this layer. */
    public Matrix a;

    /** Number of channel. */
    public int channelNum;
    /** Number of kernel. */
    public int kernelNum;
    /** Row of input. */
    public int inRow;
    /** Column of input. */
    public int inCol;
    /** Row of output. */
    public int outRow;
    /** Column of output. */
    public int outCol;
    /** Number of nodes for this layer. */
    public int nodesNum;
    /** Row of weight matrix. */
    public int wRow;
    /** Column of weight matrix. */
    public int wCol;
    /** Pooling matrix size. */
    public int poolSize;

    public void exit(String msg){
        System.out.println(msg);
        System.exit(-1);
    }

    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public Layer(){
        ;
    }

    /**
     * Construct instead of constructor for dense layer.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public void setup(int inNum, int nodesNum, AFType afType, long seed){
        ;
    }

    /**
     * Construct instead of constructor for conv layer.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public void setup(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType, long seed){
        ;
    }

    /**
     * Construct instead of constructor for max pooling layer.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public void setup(int channelNum, int[] inShape, int poolSize){
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

    public Layer clone(){
        return new Layer();
    }
}