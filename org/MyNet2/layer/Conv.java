package org.MyNet2.layer;

import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;

import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for convolution layer.
 */
public class Conv extends Layer {
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
        this.name = "Conv";
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
    public void setup(int channelNum, int kernelNum, int[] inShape, int[] wShape, AFType afType, long seed){
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
        this.b = new Matrix(kernelNum, 1, new Random(seed));
        this.delta = new Matrix(this.channelNum, this.inRow*this.inCol);

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
            case SOFTMAX:
                this.actFunc = new Softmax();
                break;
            default:
                this.exit("ERROR: The specified activation function is wrong");
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

        this.x = new Matrix(in.row, this.kernelNum * this.outRow * this.outCol);
        for (int b = 0; b < in.row; b++){
            for (int k = 0; k < this.kernelNum; k++){
                for (int i = 0; i < this.outRow; i++){
                    for (int j = 0; j < this.outCol; j++){
                        for (int c = 0; c < this.channelNum; c++){
                            for (int p = 0; p < this.wRow; p++){
                                for (int q = 0; q < this.wCol; q++){
                                    x.matrix[b][k*kMult + i*this.outCol + j] +=
                                        this.w.matrix[k][c*cWMult + p*this.wCol + q]
                                        * in.matrix[b][c*cInMult + (i+p)*this.inCol + (j+q)]
                                        + this.b.matrix[k][0];
                                }
                            }
                        }
                    }
                }
            }
        }

        this.a = this.actFunc.calc(x);

        return a.clone();
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "Convolution\nact: %s\n"
            + "%d, %d, %d => (%d, %d) => %d, %d, %d",
            this.actFuncName, this.channelNum, this.inRow, this.inCol, this.wRow, this.wCol, this.kernelNum, this.outRow, this.outCol
        );

        return str;
    }
}