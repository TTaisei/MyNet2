package org.myNet2.actFunc;

import java.lang.Math;
import org.myNet2.*;

/**
 * Softmax function.
 */
public class Softmax extends ActivationFunction {
    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public Softmax(){
        ;
    }

    /**
     * Execute this actiation function.
     * @param in linear transformationed matrix.
     * @return output matrix.
     */
    @Override
    public Matrix calc(Matrix in){
        Matrix rtn = new Matrix(in.row, in.col);
        double denominator;
        double cal;
        for (int i = 0; i < rtn.row; i++){
            denominator = 0.;
            for (int j = 0; j < rtn.col; j++){
                cal = Math.exp(in.matrix[i][j]);
                rtn.matrix[i][j] = cal;
                denominator += cal;
            }
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] /= denominator;
            }
        }

        return rtn;
    }

    /**
     * Calcurate this activation function's differential.
     * @param in Matrix of input.
     * @return The result of differentiating this activation function.
     */
    @Override
    public Matrix diff(Matrix in){
        Matrix rtn = new Matrix(in.row, in.col);
        double denominator;
        double cal;
        for (int i = 0; i < rtn.row; i++){
            denominator = 0.;
            for (int j = 0; j < rtn.col; j++){
                cal = Math.exp(in.matrix[i][j]);
                rtn.matrix[i][j] = cal;
                denominator += cal;
            }
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] /= denominator;
                rtn.matrix[i][j] = rtn.matrix[i][j] * (1-rtn.matrix[i][j]);
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        return "Softmax";
    }
}
