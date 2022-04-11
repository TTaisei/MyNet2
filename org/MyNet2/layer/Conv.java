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
    /** Number of chanel. */
    public int channelNum;
    /** Number of kernel. */
    public int kernelNum;
    /** Row of weight matrix. */
    public int wRow;
    /** Column of weight matrix. */
    public int wCol;
    /** Row of output. */
    public int outRow;
    /** Column of output. */
    public int outCol;

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inShape Shape of input.
     * @param wShape Shape of weight matrix.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType){
        this.setup(channelNum, kernelNum, inShape, wShape, 0, afType);
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
    public Conv(int channelNum, int kernelNum, int[] inShape, int[] wShape, long seed, AFType afType){
        this.setup(channelNum, kernelNum, inShape, wShape, seed, afType);
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
    protected void setup(int channelNum, int kernelNum, int[] inShape, int[] wShape, long seed, AFType afType){
        if (inShape.length != 2){
            this.exit("inShape length is wrong.");
        }else if (wShape.length != 2){
            this.exit("wShape length is wrong.");
        }

        this.channelNum = channelNum;
        this.kernelNum = kernelNum;
        this.outRow = inShape[0] - wShape[0] + 1;
        this.outCol = inShape[1] - wShape[1] + 1;

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

        Matrix rtn = new Matrix(in.row, this.kernelSize * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.kernelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        for (int c = 0; c < this.channelNum; c++){
                            for (int p = 0; p < this.wRow; p++){
                                for (int q = 0; q < this.wCol; q++){
                                    rtn.matrix[b][k*this.kernelNum + i*this.outRow + j] +=
                                        this.w.matrix[k][c*this.channelNum + p*this.wRow + q]
                                        * in.matrix[b][c*this.channelNum + (i+p)*this.wRow + (j+q)];
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