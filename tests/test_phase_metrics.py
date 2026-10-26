import unittest
import numpy as np
from screening_analysis import phase_metrics as metrics


class MetricChecks(unittest.TestCase):
    def test_known_binary_metrics_and_auc_ties(self):
        y=np.array([0,0,1,1]);p=np.array([0,1,1,1]);scores=np.array([[.9,.1],[.6,.4],[.6,.4],[.1,.9]])
        result=metrics.classification_metrics(y,p,scores)
        self.assertEqual(result['balanced_accuracy'],.75);self.assertEqual(result['auroc'],.875)
        self.assertAlmostEqual(result['matthews_correlation'],1/np.sqrt(3))
        self.assertAlmostEqual(result['f1_macro'],(.6666666666666666+.8)/2)
        self.assertEqual(result['confusion_matrix'],[[1,1],[0,2]])

    def fixture(self):
        return np.array([[0],[10],[.1],[10.1],[-.1],[9.9]]),np.array([0,1]*3),np.array([1,1,2,2,3,3])

    def test_grouped_folds_bootstrap_and_exact_permutation_refit(self):
        x,y,g=self.fixture();r=metrics.evaluate(x,y,g,bootstrap_repetitions=100)
        self.assertEqual(r['balanced_accuracy'],1);self.assertEqual(r['permutation']['permutations'],8)
        self.assertAlmostEqual(r['permutation']['pvalue'],.25)
        self.assertEqual(r['bootstrap']['independent_groups'],3)
        for fold in r['folds']:
            self.assertFalse(set(g[fold['train_indices']])&set(g[fold['test_indices']]))
        self.assertEqual(r['statistical_status'],'limited')

    def test_simultaneous_factor_and_seed_holdout(self):
        x=[];y=[];groups=[];factor=[]
        for g in range(3):
            for f in range(2):
                for label in range(2):x.append([label*10+.1*g]);y.append(label);groups.append(g);factor.append(f)
        r=metrics.grouped_predictions(x,y,groups,factor)
        self.assertEqual(r['metrics']['balanced_accuracy'],1)
        for fold in r['folds']:
            train=np.array(fold['train_indices']);test=np.array(fold['test_indices'])
            self.assertFalse(set(np.array(groups)[train])&set(np.array(groups)[test]))
            self.assertFalse(set(np.array(factor)[train])&set(np.array(factor)[test]))

    def test_multiclass_and_infinity_preservation(self):
        y=np.array(['WC','AHC','liquid','halo']*3);score=np.tile(np.eye(4),(3,1))
        classes=np.array(['WC','AHC','liquid','halo'])
        result=metrics.classification_metrics(y,y,score,classes)
        self.assertEqual(result['balanced_accuracy'],1);self.assertEqual(result['auroc'],1)
        diagrams=[(np.array([[0,np.inf]]),np.empty((0,2))),
                  (np.array([[0,np.inf],[0,np.inf]]),np.empty((0,2))),
                  (np.array([[0,np.inf]]),np.empty((0,2)))]
        r=metrics.within_between(diagrams,[0,1,0],[1,2,3])
        self.assertEqual(r['between']['infinite'],2);self.assertEqual(r['within']['finite_mean'],0)


if __name__=='__main__':unittest.main()
