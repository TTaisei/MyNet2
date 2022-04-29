package optimizer;

/**
 * Class for stochastic gradient descent.
 */
public class SGD extends Optimizer {
    /**
     * Constructor fot this class.
     */
    public SGD(){
        ;
    }

    /**
     * Doing back propagation.
     * @param x input matrix.
     * @param y Result of forward propagation.
     * @param t Answer.
     */
    protected void back(Matrix x, Matrix y, Matrix t){
        ;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        int backNum = (int)(x.row / batchSize) + 1;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            for (int j = 0; j < backNum; j++){
                y = this.net.forward(xs[j]);
                this.back(xs[j], y, ts[j]);
                System.out.printf("\rloss: %.4f", this.lossFunc.calc(y, ts[j]));
            }
            System.out.println();
        }

        return y;
    }

    /**
     * Run learning.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize,
                      Matrix valX, Matrix valT){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix[][] valxt = this.makeMiniBatch(valX, valT, batchSize, rand);
        Matrix[] valxs = valxt[0];
        Matrix[] valts = valxt[1];
        Matrix y = ts[0].clone();
        Matrix valY;
        int backNum = (int)(x.row / batchSize) + 1;

        for (int i = 0; i < nEpoch; i++){
            System.out.printf("Epoch %d/%d\n", i+1, nEpoch);
            for (int j = 0; j < backNum; j++){
                valY = this.net.forward(valxs[j]);
                y = this.net.forward(xs[j]);
                this.back(xs[j], y, ts[j]);
                System.out.printf(
                    "\rloss: %.4f - valLoss: %.4f",
                    this.lossFunc.calc(y, ts[j]),
                    this.lossFunc.calc(valY, valts[j])
                );
            }
            System.out.println();
        }

        return y;
    }

    /**
     * Run learning and save log.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize, String fileName){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        int backNum = (int)(x.row / batchSize) + 1;
        double loss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss\n");
            for (int i = 0; i < nEpoch; i++){
                for (int j = 0; j < backNum; j++){
                    y = this.net.forward(xs[j]);
                    this.back(xs[j], y, ts[j]);
                }
                loss = this.lossFunc.calc(y, ts[ts.length-1]);
                fp.printf("%d,%f\n", i+1, loss);
            }
        }catch (IOException e){
            System.out.println("IO Exception");
            System.exit(-1);
        }

        return y;
    }

    /**
     * Run learning and save log.
     * @param x Input layer.
     * @param t Answer.
     * @param nEpoch Number of epoch.
     * @param batchSize Size of batch.
     * @param valX Input layer for validation.
     * @param valT Answer for validation.
     * @param fileName Name of logging file.
     * @return Output of this network.
     */
    public Matrix fit(Matrix x, Matrix t, int nEpoch, int batchSize,
                      Matrix valX, Matrix valT, String fileName){
        Matrix[][] xt = this.makeMiniBatch(x, t, batchSize, rand);
        Matrix[] xs = xt[0];
        Matrix[] ts = xt[1];
        Matrix y = ts[0].clone();
        Matrix valY;
        int backNum = (int)(x.row / batchSize) + 1;
        double loss = 0., valLoss = 0.;

        try(
            PrintWriter fp = new PrintWriter(fileName);
        ){
            fp.write("Epoch,loss,valLoss\n");
            for (int i = 0; i < nEpoch; i++){
                for (int j = 0; j < backNum; j++){
                    y = this.net.forward(xs[j]);
                    this.back(xs[j], y, ts[j]);
                }
                valY = this.net.forward(valX);
                loss = this.lossFunc.calc(y, ts[ts.length-1]);
                valLoss = this.lossFunc.calc(valY, valT);
                fp.printf("%d,%f,%f\n", i+1, loss, valLoss);
            }
        }catch (IOException e){
            System.out.println("IO Exception");
            System.exit(-1);
        }

        return y;
    }
}
