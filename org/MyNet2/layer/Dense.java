package org.MyNet2.layer;

import java.util.Random;
import org.MyNet2.*;
import org.MyNet2.actFunc.*;

public class Dense extends Layer {
    /** The list of weight of this layer. */
    public Matrix w;
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
    public Dense(int nodesNum, AFType afType){
        this.name = "Dense";
        this.nodesNum = nodesNum;
        this.afType = afType;
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
    protected void setup(int inNum, int nodesNum, AFType afType, long seed){
        this.name = "Dense";
        this.inNum = inNum + 1;
        this.nodesNum = nodesNum;
        
        // this.w = new Matrix(this.inNum, nodesNum, new Random(seed), -1, 1);
        this.w = new Matrix(this.inNum, nodesNum, 0.5);
        this.x = new Matrix(this.inNum, this.nodesNum);
        this.a = new Matrix(this.inNum, this.nodesNum);

        switch(afType) {
        case SIGMOID:
            this.actFunc = new Sigmoid();
            break;
        case RELU:
            this.actFunc = new ReLu();
            break;
        case TANH:
            this.actFunc = new Tanh();
            break;
        case LINER:
            this.actFunc = new Liner();
            break;
        default:
            System.out.println("ERROR: The specified activation function is wrong");
            System.exit(-1);
        }
        this.actFuncName = this.actFunc.toString();
    }

    @Override
    public Matrix forward(Matrix in){
        Matrix in_ = in.appendCol(1.0);
        return this.actFunc.calc(in_.dot(this.w));
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "Dense\n"
            + "nodes num: %d, activation function: %s", this.nodesNum, this.actFuncName
        );

        return str;
    }
}