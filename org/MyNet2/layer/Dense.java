package org.MyNet2.layer;

import java.util.Random;
import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for dense layer.
 */
public class Dense extends Layer {
    /** The list of weight for this layer. */
    public Matrix w;
    /** Activation function of this layer. */
    public ActivationFunction actFunc;
    /** Name of this layer's activation function. */
    public String actFuncName;

    /** Number of inputs contain bias. */
    public int inNum;

    /**
     * Constructor for this class.
     * @param nodesNum Number of nodes.
     * @param afType Type of activation function.
     */
    public Dense(int nodesNum, AFType afType){
        this.name = "Dense";
        this.nodesNum = nodesNum;
        this.afType = afType;
        this.delta = new Matrix(nodesNum, 1);
    }

    /**
     * Constructor for this class.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     */
    public Dense(int inNum, int nodesNum, AFType afType){
        this.setup(inNum, nodesNum, afType, 0);
    }

    /**
     * Constructor for this class.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public Dense(int inNum, int nodesNum, AFType afType, long seed){
        this.setup(inNum, nodesNum, afType, seed);
    }

    /**
     * Construct instead of constructor.
     * @param inNum Number of inputs don't contain bias.
     * @param nodesNum Number of nodes of this class.
     * @param afType Type of activation function for this layer.
     * @param seed Number of seed for random class.
     */
    public void setup(int inNum, int nodesNum, AFType afType, long seed){
        this.name = "Dense";
        this.inNum = inNum + 1;
        this.nodesNum = nodesNum;
        
        this.w = new Matrix(this.inNum, nodesNum, new Random(seed), -1, 1);
        this.x = new Matrix(this.inNum, this.nodesNum);
        this.a = new Matrix(this.inNum, this.nodesNum);
        this.delta = new Matrix(nodesNum, 1);

        switch(afType) {
        case SIGMOID:
            this.actFunc = new Sigmoid();
            break;
        case RELU:
            this.actFunc = new ReLU();
            break;
        case TANH:
            this.actFunc = new Tanh();
            break;
        case LINEAR:
            this.actFunc = new Linear();
            break;
        case SOFTMAX:
            this.actFunc = new Softmax();
            break;
        default:
            System.out.println("ERROR: The specified activation function is wrong");
            System.exit(-1);
        }
        this.actFuncName = this.actFunc.toString();
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    @Override
    public Matrix forward(Matrix in){
        this.x = in.appendCol(1.0).dot(this.w);
        this.a = this.actFunc.calc(this.x);
        return this.a.clone();
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "Dense\nact: %s\n"
            + "%d => %d", this.actFuncName, this.inNum-1, this.nodesNum
        );

        return str;
    }
}