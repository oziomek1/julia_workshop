#=
Based on https://github.com/denizyuret/Knet.jl
=#
# Boston housing
using Knet

predict(w, x) = w[1]*x.+ w[2]
loss(w,x,y) = mean(abs2, y-predict(w,x))
lossgrad = grad(loss)

function train(w, data; lr=0.05)
    for (x,y) in data
        dw = lossgrad(w,x,y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end

include(Knet.dir("data", "housing.jl"))
x, y = housing()
w = Any[ 0.1*randn(1,13), 0.0 ]
for i=1:25; train(w, [(x,y)]); println(loss(w,x,y)); end

# MNIST

predict(w,x) = w[1]*mat(x) .+ w[2]
#= mat is for conversion 28x28x1 x N -> 784 x N =#
loss(w,x,y) = nll(predict(w,x), y)
#= nll is for conversion to 10 x N matric of output =#
lossgrad = grad(loss)

include(Knet.dir("data", "mnist.jl"))
X_train, y_train, X_test, y_test = mnist()

d_train = minibatch(X_train, y_train, 100)
d_test = minibatch(X_test, y_test, 100)

w = Any[ 0.1f0*randn(Float32, 10, 784), zeros(Float32,10,1) ]
println((:epoch, 0, :trn, accuracy(w,d_train,predict), :tst, accuracy(w, d_test, predict)))

for epoch=1:10
    train(w, d_train, lr=0.1)
    println((:epoch, epoch, :trn, accuracy(w, d_train, predict), :tst, accuracy(w, d_test, predict)))
end

# Multilayer perceptron

function predict(w,x)
    x = mat(x)
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+w[end]
end

w = Any[ 0.1f0*randn(Float32,64,784), zeros(Float32,64,1),
         0.1f0*randn(Float32,10,64),  zeros(Float32,10,1) ]

function train(model, data, optimizer)
    for (x,y) in data
         grads = lossgrad(model, x,y)
         update!(model, grads, optimizer)
     end
end

o = optimizers(w, Adam)
println((:epoch, 0, :trn, accuracy(w,d_train,predict), :tst, accuracy(w,d_test,predict)))
for epoch=1:25
    train(w, d_train, o)
    println((:epoch, epoch, :trn, accuracy(w,d_train, predict), :tst, accuracy(w,d_test,predict)))
end

# CNN with LeNet model

function predict(w, input)
    x1 = pool(relu.(conv4(w[1], input) .+ w[2]))
    x2 = pool(relu.(conv4(w[3], x1) .+ w[4]))
    x3 = relu.(w[5]*mat(x2) .+ w[6])
    return w[7]*x3 .+ w[8]
end

w = Any[ xavier(Float32,5,5,1,20),  zeros(Float32,1,1,20,1),
         xavier(Float32,5,5,20,50), zeros(Float32,1,1,50,1),
         xavier(Float32,500,800),   zeros(Float32,500,1),
         xavier(Float32,10,500),    zeros(Float32,10,1) ]

d_train = minibatch(X_train,y_train,100,xtype=KnetArray)
d_test = minibatch(X_test,y_test,100,xtype=KnetArray)
w = map(KnetArray, w)

# COMMENTED OUT BECAUSE OF NEED OF CUDA FOR CONVOLUTION
# o = optimizers(w, Adam)
# println((:epoch, 0, :trn, accuracy(w,d_train,predict), :tst, accuracy(w,d_test,predict)))
# for epoch=1:10
#     train(w, d_train, o)
#     println((:epoch, epoch, :trn, accuracy(w,d_train, predict), :tst, accuracy(w,d_test,predict)))
# end
