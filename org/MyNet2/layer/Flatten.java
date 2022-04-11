package org.MyNet2.layer;

import org.MyNet2.*;

public class Flatten extends Layer {
    /** Constructor for this class. */
    public Flatten(){
        ;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    public Matrix forward(Matrix4d in){
        Matrix rtn = new Matrix(in.shape[0], in.shape[1]*in.shape[2]*in.shape[3]);

        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < in.shape[1]; j++){
                for (int k = 0; k < in.shape[2]; k++){
                    for (int l = 0; l < in.shape[3]; l++){
                        rtn.matrix[i][j*in.shape[1]+k*in.shape[2]+l] = in.matrix.get(i).matrix.get(j).matrix[k][l];
                    }
                }
            }
        }

        return rtn;
    }

    /**
     * Doing back propagation.
     */
    @Override
    public void back(){
        ;
    }
}