package org.MyNet2.layer;

import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for convolution layer.
 */
public class Conv extends Layer {
    /** The list of weight for this layer */
    public Matrix w;
    /** Type of activation function for this layer. */
    public AFType afType;
    /** Activation function of this layer. */
    public ActivationFunction actFunc;
    /** Name of this layer's activation function. */
    public String actFuncName;

    /** Number of channel. */
    public int channelNum;
    /** Number of kernel. */
    public int kernelNum;
    /** Row of weight matrix. */
    public int wRow;
    /** Column of weight matrix. */
    public int wCol;
    /** Row of input. */
    public int inRow;
    /** Column of input. */
    public int inCol;
    /** Row of output. */
    public int outRow;
    /** Column of output. */
    public int outCol;

    /**
     * Constructor for this class.
     * @param kernelNum Number of kernel.
     * @param wShape Shape of weight matrix.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int kernelNum, int[] wShape, AFType afType){
        if (wShape.length != 2){
            this.exit("wShape length is wrong");
        }
        this.kernelNum = kernelNum;
        this.wRow = wShape[0];
        this.wCol = wShape[1];
        this.afType = afType;
    }

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inShape Shape of input.
     * @param wShape Shape of weight matrix.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType){
        this.setup(channelNum, kernelNum, inShape, wShape, afType, 0);
    }

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inShape Shape of input.
     * @param wShape Shape of weight matrix.
     * @param seed Number of seed for random class.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType, long seed){
        this.setup(channelNum, kernelNum, inShape, wShape, afType, seed);
    }

    /**
     * Construct instead of constructor.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inShape Shape of input.
     * @param wShape Shape of weight matrix.
     * @param seed Number of seed for random class.
     * @param afType Type of activation fucntion for this layer.
     */
    protected void setup(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType, long seed){
        if (inShape.length != 2){
            this.exit("inShape length is wrong.");
        }else if (wShape.length != 2){
            this.exit("wShape length is wrong.");
        }
        this.name = "Conv";

        this.channelNum = channelNum;
        this.kernelNum = kernelNum;
        this.inRow = inShape[0];
        this.inCol = inShape[1];
        this.outRow = inShape[0] - wShape[0] + 1;
        this.outCol = inShape[1] - wShape[1] + 1;
        this.wRow = wShape[0];
        this.wCol = wShape[1];

        this.w = new Matrix(kernelNum, channelNum * wRow * wCol, new Random(seed));

        this.afType = afType;
        switch (afType){
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
        int kMult = this.outRow * this.outCol;
        int cWMult = this.wRow * this.wCol;
        int cInMult = this.inRow * this.inCol;

        Matrix rtn = new Matrix(in.row, this.kernelNum * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.kernelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        for (int c = 0; c < this.channelNum; c++){
                            for (int p = 0; p < this.wRow; p++){
                                for (int q = 0; q < this.wCol; q++){
                                    rtn.matrix[b][k*kMult + i*this.outCol + j] +=
                                        this.w.matrix[k][c*cWMult + p*this.wCol + q]
                                        * in.matrix[b][c*cInMult + (i+p)*this.inCol + (j+q)];
                                }
                            }
                        }
                    }
                }
            }
        }

        return this.actFunc.calc(rtn);
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "Convolution\n"
            + "channels: %d, kernels: %d, conv size: %dx%d",
            this.channelNum, this.kernelNum, this.wRow, this.wCol
        );

        return str;
    }
}