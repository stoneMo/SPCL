from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster

def visual_cluster(feature, label, args, epoch):

    # input feature: (N, 512)
    # input label: (N, )

    tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(feature)

    fig_name = args.figs_dir+'/repre_epoch_'+str(epoch+1)

    cluster.tsneplot(score=tsne_em, colorlist=label, legendpos='upper right', legendanchor=(1.15, 1), figname=fig_name)


