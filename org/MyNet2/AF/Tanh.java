package org.MyNet.af;

import java.lang.Math;
import org.MyNet2.*;

/**
 * Tanh function.
 */
public class Tanh extends ActivationFunction {
    /**
     * Constructor for this class.
     * Nothing to do.
     */
    public Tanh(){
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
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = Math.tanh(in.matrix[i][j]);
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
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = 1 - Math.pow(Math.tanh(in.matrix[i][j]), 2);
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        return "Tanh";
    }
}