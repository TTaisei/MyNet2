package org.MyNet2;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Random;

/**
 * Class for two dimentional matrix.
 */
public class Matrix {
    /** Value of this matrix. */
    public double[][] matrix;
    /** Row and col of this matrix. */
    public int row, col;

    /**
     * Exit this program.
     * @param msg Message for printing.
     */
    protected void exit(String msg){
        System.out.println(msg);
        System.exit(-1);
    }

    /**
     * Constructor for this class.
     * @param row Number of row.
     * @param col Number of col.
     */
    public Matrix(int row, int col){
        this.row = row;
        this.col = col;
        this.matrix = new double[this.row][this.col];
    }

    /**
     * Constructor for this class.
     * @param row Number of row.
     * @param col Number of col.
     * @param num Number to fill.
     */
    public Matrix(int row, int col, double num){
        this.row = row;
        this.col = col;

        this.matrix = new double[this.row][this.col];
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                this.matrix[i][j] = num;
            }
        }
    }

    /**
     * Constructor for this class.
     * @param row Number of row.
     * @param col Number of col.
     * @param rand Random instance.
     */
    public Matrix(int row, int col, Random rand){
        this.row = row;
        this.col = col;

        this.matrix = new double[this.row][this.col];
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                this.matrix[i][j] = rand.nextDouble();
            }
        }
    }

    /**
     * Constructor for this class.
     * @param row Number of row.
     * @param col Number of col.
     * @param rand Random instance.
     * @param min Number of min for range.
     * @param max Number of max for range.
     */
    public Matrix(int row, int col, Random rand, double min, double max){
        this.row = row;
        this.col = col;

        this.matrix = new double[this.row][this.col];
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                this.matrix[i][j] = rand.nextDouble() * (max-min) + min;
            }
        }
    }

    /**
     * Constructor for this class.
     * @param in Two dimentional matrix of type double[][].
     */
    public Matrix(double[][] in){
        this.row = in.length;
        this.col = in[0].length;

        this.matrix = new double[this.row][this.col];
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                this.matrix[i][j] = in[i][j];
            }
        }
    }

    /**
     * Constructor for this class.
     * @param in Two dimentional matrix of type ArrayList<ArrayList<Double>>.
     */
    public Matrix(ArrayList<ArrayList<Double>> in){
        this.row = in.size();
        this.col = in.get(0).size();

        this.matrix = new double[this.row][this.col];
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                this.matrix[i][j] = in.get(i).get(j);
            }
        }
    }

    /**
     * Add a matrix to this matrix.
     * @param matrix Append matrix.
     * @return New matrix instance.
     */
    public Matrix add(Matrix matrix){
        if (matrix.row != this.row || matrix.col != this.col){
            this.exit("Adding error");
        }

        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[i][j] + matrix.matrix[i][j];
            }
        }

        return rtn;
    }

    /**
     * Add a number to this matrix.
     * @param num Append number.
     * @return New matrix instance.
     */
    public Matrix add(double num){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[i][j] + num;
            }
        }

        return rtn;
    }

    /**
     * Subtract a matrix from this matrix.
     * @param matrix Matrix to subtract.
     * @return New matrix instance.
     */
    public Matrix sub(Matrix matrix){
        if (matrix.row != this.row || matrix.col != this.col){
            this.exit("Subtracting error");
        }

        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[i][j] - matrix.matrix[i][j];
            }
        }

        return rtn;
    }

    /**
     * Subtract a number from this matrix.
     * @param num Number to subtract.
     * @return New matrix instance.
     */
    public Matrix sub(double num){
        return this.add(-num);
    }

    /**
     * Multiply this matrix by a number.
     * @param num Multiplier.
     * @return New matrix instance.
     */
    public Matrix mult(double num){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[i][j] * num;
            }
        }

        return rtn;
    }

    /**
     * Divid this matrix by a number.
     * @param num Divider.
     * @return New matrix instance.
     */
    public Matrix div(double num){
        return this.mult(1 / num);
    }

    /**
     * Dot product for two matrices.
     * @param matrix Matrix to dot product.
     * @return New dot producted Matrix instance.
     */
    public Matrix dot(Matrix matrix){
        if (this.col != matrix.row){
            this.exit("dot producting error");
        }

        Matrix rtn = new Matrix(this.row, matrix.col);
        double num = 0;
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < matrix.col; j++){
                num = 0;
                for (int k = 0; k < this.col; k++){
                    num += this.matrix[i][k] * matrix.matrix[k][j];
                }
                rtn.matrix[i][j] = num;
            }
        }

        return rtn;
    }

    /**
     * Create transpose of this matrix.
     * @return New matrix instance transposed of this matrix.
     */
    public Matrix T(){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[j][i] = this.matrix[i][j];
            }
        }

        return rtn;
    }

    /**
     * Fill this matrix with a number.
     * @param num Number to fill.
     * @return New matrix instance.
     */
    public Matrix fillNum(double num){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = num;
            }
        }

        return rtn;
    }

    /**
     * Fill this matrix with random numbers has range 0~1.
     * @return New matrix instance.
     */
    public Matrix fillNextRandom(){
        return this.fillNextRandom(0);
    }

    /**
     * Fill this matrix with random numbers has range 0~1.
     * @param seed Number of seed.
     * @return New matrix instance.
     */
    public Matrix fillNextRandom(long seed){
        Random rand = new Random(seed);
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = rand.nextDouble();
            }
        }

        return rtn;
    }

    /**
     * Fill this matrix with random number has range min~max.
     * @param min Number of min for range.
     * @param max Number of max for range.
     */
    public Matrix fillRandom(double min, double max){
        return this.fillRandom(min, max, 0);
    }

    /**
     * Fill this matrix with random number has range min~max.
     * @param min Number of min for range.
     * @param max Number of max for range.
     * @param seed Number of seed.
     */
    public Matrix fillRandom(double min, double max, long seed){
        Random rand = new Random(seed);
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = rand.nextDouble()*(max-min) + min;
            }
        }

        return rtn;
    }

    /**
     * Append a number to the side of this matrix.
     * @param num Number to append.
     * @return New matrix instance.
     */
    public Matrix appendCol(double num){
        Matrix rtn = new Matrix(this.row, this.col+1);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[i][j];
            }
        }
        for (int i = 0; i < this.row; i++){
            rtn.matrix[i][this.col] = num;
        }

        return rtn;
    }

    /**
     * Stack matrices horizontally.
     * @param matrices Matrices to stack.
     *                 These should not have more than two columns.
     * @return New Matrix instance stacked.
     */
    public static Matrix hstack(Matrix ... matrices){
        Matrix rtn = new Matrix(matrices[0].row, matrices.length);
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = matrices[j].matrix[i][0];
            }
        }

        return rtn;
    }

    /**
     * Stack matrices vertical.
     * @param matrices Matrices to stack.
     *                 These should not have more than two rows.
     * @return New Matrix instance stacked.
     */
    public static Matrix vstack(Matrix ... matrices){
        Matrix rtn = new Matrix(matrices.length, matrices[0].col);
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = matrices[i].matrix[0][j];
            }
        }

        return rtn;
    }

    /**
     * Split this matrix vertically.
     * @param num Number of split.
     * @return Array of Matrix instance.
     */
    public Matrix[] vsplit(int num){
        Matrix[] rtn = new Matrix[num];
        int size = this.row / num;
        if (size * num != this.row){
            this.exit("vsplit error");
        }

        for (int i = 0; i < num; i++){
            rtn[i] = new Matrix(size, this.col);
            int row = i * size;
            for (int j = 0; j < size; j++){
                for (int k = 0; k < this.col; k++){
                    rtn[i].matrix[j][k] = this.matrix[row + j][k];
                }
            }
        }

        return rtn;
    }

    /**
     * Sort this matrix vertically.
     * @param order Order of sort.
     * @return Matrix instance.
     */
    public Matrix vsort(int[] order){
        Matrix rtn = new Matrix(order.length, this.col);

        for (int i = 0; i < order.length; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = this.matrix[order[i]][j];
            }
        }

        return rtn;
    }

    /**
     * Return absolute value of this matrix.
     * @return New matrix instance.
     */
    public Matrix abs(){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = Math.abs(this.matrix[i][j]);
            }
        }

        return rtn;
    }

    /**
     * Calcurate average of each columns.
     * @return Matrix instance that had everage of each columns in this matrix.
     */
    public Matrix meanCol(){
        Matrix rtn = new Matrix(1, this.col);

        double num = 0.;
        for (int j = 0; j < this.col; j++){
            num = 0;
            for (int i = 0; i < this.row; i++){
                num += this.matrix[i][j];
            }
            rtn.matrix[0][j] = num / this.row;
        }

        return rtn;
    }

    /**
     * Calcurate average of each rows.
     * @return Matrix instance that had everage of each rows in this matrix.
     */
    public Matrix meanRow(){
        Matrix rtn = new Matrix(this.row, 1);

        double num = 0.;
        for (int i = 0; i < this.row; i++){
            num = 0;
            for (int j = 0; j < this.row; j++){
                num += this.matrix[i][j];
            }
            rtn.matrix[i][0] = num / this.col;
        }

        return rtn;
    }

    /**
     * Calucurate square root each number of this matrix.
     * @return New matrix instance.
     */
    public Matrix sqrt(){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                rtn.matrix[i][j] = Math.sqrt(this.matrix[i][j]);
            }
        }

        return rtn;
    }

    /**
     * Power of a matrix element.
     * @return Multiplying a matrix by itself.
     */
    public Matrix pow(){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = Math.pow(this.matrix[i][j], 2);
            }
        }

        return rtn;
    }

    /**
     * Power of a matrix element.
     * @param num Number to power.
     * @return Multiplying a matrix by itself.
     */
    public Matrix pow(int num){
        Matrix rtn = new Matrix(this.row, this.col);
        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < rtn.col; j++){
                rtn.matrix[i][j] = Math.pow(this.matrix[i][j], num);
            }
        }

        return rtn;
    }

    /**
     * Make a 4 dimentional matrix from this 2 dimentional matrix.
     * @param shape Shape of 4 dimentional matrix.
     * @return 2 dimentional matrix.
     */
    public Matrix4d toMatrix4d(int[] shape){
        if (shape.length != 4){
            this.exit("shape is wrong.");
        }else if(shape[0] != this.row){
            this.exit("row number is wrong.");
        }
        int jMult = shape[2] * shape[3];

        Matrix4d rtn = new Matrix4d(shape);
        for (int i = 0; i < shape[0]; i++){
            for (int j = 0; j < shape[1]; j++){
                for (int k = 0; k < shape[2]; k++){
                    for (int l = 0; l < shape[3]; l++){
                        rtn.matrix.get(i).matrix.get(j).matrix[k][l] = this.matrix[i][j*jMult + k*shape[3] + l];
                    }
                }
            }
        }

        return rtn;
    }

    /**
     * Make a 4 dimentional matrix from this 2 dimentional matrix.
     * @param shape0 Shape of 4 dimentional matrix.
     * @param shape1 Shape of 4 dimentional matrix.
     * @param shape2 Shape of 4 dimentional matrix.
     * @param shape3 Shape of 4 dimentional matrix.
     * @return 2 dimentional matrix.
     */
    public Matrix4d toMatrix4d(int shape0, int shape1, int shape2, int shape3){
        if(shape0 != this.row){
            this.exit("row number is wrong.");
        }
        int jMult = shape2 * shape3;

        Matrix4d rtn = new Matrix4d(new int[]{shape0, shape1, shape2, shape3});
        for (int i = 0; i < shape0; i++){
            for (int j = 0; j < shape1; j++){
                for (int k = 0; k < shape2; k++){
                    for (int l = 0; l < shape3; l++){
                        rtn.matrix.get(i).matrix.get(j).matrix[k][l] = this.matrix[i][j*jMult + k*shape3 + l];
                    }
                }
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        String str = "[";

        int i = 0;
        for (double[] ele: matrix){
            if (i == 0){
                str += "[";
            }else{
                str += "\n [";
            }
            i++;
            for (double num: ele){
                if (num < 0){
                    str += String.format("%.4f ", num);
                }else{
                    str += String.format(" %.4f ", num);
                }
            }
            str += "]";
        }
        str += "]\n";

        return str;
    }

    /**
     * Method to compare this Matrix instance and a Matrix instance.
     * Without override.
     * @param o A Matrix instance.
     * @return Is equal?
     */
    public boolean equals(Matrix o){
        if (o == this){
            return true;
        }
        if (this.row != o.row || this.col != o.col){
            return false;
        }

        for (int i = 0; i < this.row; i++){
            for (int j = 0; j < this.col; j++){
                if (this.matrix[i][j] != o.matrix[i][j]){
                    return false;
                }
            }
        }

        return true;
    }

    @Override
    public Matrix clone(){
        return new Matrix(this.matrix);
    }

    @Override
    public int hashCode(){
        return (int)this.matrix[0][0];
    }
}