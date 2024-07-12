#include "regression.h"

Regression::Regression(uint64_t D){
    d = D;
    weights = zeros(d,1);
    weights_ = zeros(d,1);
    bias = 0;
    epsilon = EPS;
    eta = ETA;
}

matrix Regression::transform(matrix X){
    uint64_t n = X.shape().first;
    matrix PHI(n,d);
    for (uint64_t i = 0 ; i < n ; i++){
        PHI(i,0) = X(i,0); 
        for (uint64_t j = 1 ; j  < d ; j++){
            PHI(i,j) = PHI(i,j - 1)*PHI(i,0);
        }
    }
    return PHI;
}

double Regression::l2loss(matrix X, matrix Y){
    matrix Y_pred = matmul(X,weights) + bias;
    __size d1 = Y.shape(), d2 = Y_pred.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot compute loss of vectors with dimensions ( "+std::to_string(d1.first)+" , "
        +std::to_string(d1.second)+" ) and ( "+std::to_string(d2.first)+" , "+std::to_string(d2.second)+" ) do not match");
    }
    uint64_t n = max(d1.first,d1.second);
    double loss = norm(Y_pred - Y);
    loss *= loss;
    loss /= n;
    return loss;
}

pair<matrix, double> Regression::l2lossDerivative(matrix X, matrix Y){
    matrix Y_pred = matmul(X,weights) + bias;
    __size d1 = Y.shape(), d2 = Y_pred.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot compute loss derivative of vectors with dimensions ( "+std::to_string(d1.first)+" , "
        +std::to_string(d1.second)+" ) and ( "+std::to_string(d2.first)+" , "+std::to_string(d2.second)+" ) do not match");
    }
    uint64_t n = X.shape().first;
    matrix Xt = X.transpose();
    matrix dw = 2*(matmul(Xt,Y_pred - Y))/n;
    double db = 2*(dot(ones(1,n),Y_pred - Y))/n;
    return {dw,db};
}

matrix Regression::predict(matrix X){
    return matmul(X,weights) + bias;
}  

void Regression::GD(matrix X, matrix Y,double learning_rate, uint64_t limit){
    eta = learning_rate;
    double old_loss = 0,loss = l2loss(X,Y);
    train_loss.PB(loss);
    uint64_t iteration = 0;
    max_iterations = limit;
    while (fabs(loss - old_loss) > epsilon && iteration < max_iterations){
        old_loss = loss;
        auto[dw,db] = l2lossDerivative(X,Y);
        weights = weights - eta*dw;
        bias = bias - eta*db;
        loss = l2loss(X,Y);
        train_loss.PB(loss);
        iteration++;
    }
}

void Regression::train(matrix X,matrix Y,double learning_rate, uint64_t limit){
    matrix PHI = transform(X);
    matrix PHIt = PHI.transpose();
    matrix Z = matmul(PHIt,PHI);
    weights_ = matmul((Z.inverse()),matmul(PHIt,Y));
    GD(PHI,Y,learning_rate,limit);
    cout << "Training Loss\n";
    for (uint64_t i = 0; i < train_loss.size() ; i++){
        cout << train_loss[i] << "\n";
    }
}

void Regression::test(matrix X,matrix Y){
    matrix PHI = transform(X);
    matrix Y_pred = predict(PHI);
    uint64_t n = PHI.shape().first;
    matrix Y_closed = matmul(PHI,weights_);
    cout << "Predictions(GD) \t Predictions(C) \t True Value\n"; 
    for (uint64_t i = 0 ; i < n ; i++){
        cout << Y_pred(i,0) << "\t\t\t "<< Y_closed(i,0) << "\t\t\t " << Y(i,0) << "\n";
    }
    cout << "Testing loss " << l2loss(PHI,Y) << "\n";
    cout << "Testing accuracy " << accuracy(Y_pred,Y) << "\n";

}

double Regression::accuracy(matrix Y_pred, matrix Y){
    double acc = 0;
    uint64_t n = Y.shape().first;
    matrix diff = (Y_pred - Y)/Y;
    acc = dot(ones(1,n),1 - fabs(diff))/n;
    return acc;
}

pair<pair<matrix, matrix>, pair<matrix, matrix>> test_train_split(matrix X, matrix Y, float ratio) {
    if (ratio < 0 || ratio > 1) {
        throw std::invalid_argument("Ratio must be between 0 and 1");
    }

    uint64_t n = X.shape().first;
    uint64_t train_size = static_cast<uint64_t>(n * ratio);
    uint64_t test_size = n - train_size;

    std::vector<uint64_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    matrix X_train(train_size, X.shape().second);
    matrix Y_train(train_size, 1);
    matrix X_test(test_size, X.shape().second);
    matrix Y_test(test_size, 1);

    for (uint64_t i = 0; i < train_size; ++i) {
        for (uint64_t j = 0; j < X.shape().second; ++j) {
            X_train(i, j) = X(indices[i], j);
        }
        Y_train(i, 0) = Y(indices[i], 0);
    }

    for (uint64_t i = 0; i < test_size; ++i) {
        for (uint64_t j = 0; j < X.shape().second; ++j) {
            X_test(i, j) = X(indices[train_size + i], j);
        }
        Y_test(i, 0) = Y(indices[train_size + i], 0);
    }

    return {{X_train, Y_train}, {X_test, Y_test}};
}