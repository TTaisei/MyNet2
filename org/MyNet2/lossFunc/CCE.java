package org.myNet2.lossFunc;

import java.lang.Math;
import org.myNet2.*;

/**
 * Class for loss function.
 */
public class CCE extends LossFunction {
    /**
     * Constructor for this class.
     */
    public CCE(){
        ;
    }

    /**
     * Calcurate this loss function.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return Diiference between y and b.
     */
    @Override
    public double calc(Matrix y, Matrix t){
        double rtn = 0.;

        for (int i = 0; i < y.row; i++){
            for (int j = 0; j < y.col; j++){
                rtn += -t.matrix[i][j] * Math.log(y.matrix[i][j]);
            }
        }

        return rtn / y.row;
    }

    /**
     * Calcurate this loss function's differential.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return The result of differentialting the difference between y and t.
     */
    @Override
    public Matrix diff(Matrix y, Matrix t){
        Matrix rtn = new Matrix(y.row, y.col);

        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = -t.matrix[i][j] / y.matrix[i][j];
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        return "CategoricalCrossEntropy";
    }
}
