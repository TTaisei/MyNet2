package org.MyNet2.layer;

import java.util.Random;
import org.MyNet2.*;

/**
 * Class for layer.
 */
public class Layer {
    /** The list of weight of this layer. */
    public Matrix w;
    /** Type of activation function for this layer. */
    public actFunc.AFType afType;
    /** Activation function of this layer. */
    public actFunc.ActivationFunction actFunc;
    /** Name of this layer's activation function. */
    public String actFuncName;
    /** Number of inputs contain bias. */
    public int inNum;
    /** Number of nodes of this class. */
    public int nodesNum;
    /** Liner transformed matrix. */
    public Matrix x;
    /** Matrix of output from this layer. */
    public Matrix a;

    /**
     * Constructor for this class.
     * @param nodesNum Number of nodes.
     * @param afType Type of activation function.
     */
    public Layer(int nodesNum, actFunc.AFType afType){
        this.nodesNum = nodesNum;
        this.afType = afType;
    }

    /**
     * Constructor for this class.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     */
    public Layer(int inNum, int nodesNum, actFunc.AFType afType){
        this.Layer(inNum, nodesNum, afType, 0);
    }

    /**
     * Constructor for this class.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public Layer(int inNum, int nodesNum, actFunc.AFType afType, long seed){
        this.inNum = inNum + 1;
        this.nodesNum = nodesNum;
        
        this.w = new Matrix(inNum, nodesNum, new Random(seed));
        this.x = new Matrix(inNum, this.nodesNum);
        this.a = new Matrix(inNum, this.nodesNum);

        switch(afType) {
        case actFunc.AFType.SIGMOID:
            this.actFunc = new actFunc.Sigmoid();
            break;
        case actFunc.AFType.RELU:
            this.actFunc = new actFunc.ReLu();
            break;
        case actFunc.AFType.TANH:
            this.actFunc = new actFunc.Tanh();
            break;
        case actFunc.AFType.LINER:
            this.actFunc = new actFunc.Liner()
            break;
        default:
            ;
        }
        this.actFuncName = this.actFunc.toString();
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
        String str = String.format(
            "nodes num: %d, activation function: %s", this.nodes_num, this.actFuncName
        );

        return str;
    }
}