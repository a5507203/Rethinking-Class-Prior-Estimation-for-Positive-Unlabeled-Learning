function D = generateSNDist(alphas)
%generates  skew normal distributions mixtures and components
mus=2:2:4;
nummus=length(mus);
lambdas=[10,-10];
numlambdas=length(lambdas);
sigmas= [1,3];
numSigmas=length(sigmas);
numalphas = length(alphas);

for alph = 1:numalphas
    real_alpha = alphas(alph);
    weights = [real_alpha (1-real_alpha)];
        
    for mu = 1:nummus
        locs = [1,mus(mu)];
        for sig = 1:numSigmas
            scales=[1,sigmas(sig)];
            for lam = 1:numlambdas
                shapes = [1,lambdas(lam)];
                p1=SNDistribution(locs(1),scales(1),shapes(1));
                p0=SNDistribution(locs(2),scales(2),shapes(2));
                p=mixture(weights, {p1,p0});
                D{mu,sig,lam,alph}={p,p1,p0};
            end
        end   
    end
end
end



