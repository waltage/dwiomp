import numpy as np
import pandas as pd

from enum import Enum
import statsmodels.api as sm

class VarType(Enum):
    Log = 1
    Unit = 2
    Class = 3
    Flag = 4

class Model(object):
    def __init__(self, name):
        self.name = name
        self.ds = pd.DataFrame()
        self.cols = 0
        self.metas = list()
        self.subjects = list()
        self.depvar = None
        self.vars = dict()

        self.means = pd.DataFrame()
        self.model_df = pd.DataFrame()
        self.results = dict(
            summary=None,
            coefs=dict(),
            mse=None,
            resid=None,
            pred=None,
        )
        self.decomp_df = pd.DataFrame()

    def _rename_var(self, name, type=None, bucket=None, detail=None):
        _name = name.replace(" ", "").lower()
        varname = ""
        if type:
            varname += "{}|".format(str(type).split(".")[1])
        varname += "{}".format(_name)
        if bucket:
            varname += "|"+bucket.upper()
            if detail:
                varname += "|"+detail.upper()

        return varname

    def _add_series(self, var, series):
        self.ds.insert(self.cols, var, series)
        self.cols += 1

    @staticmethod
    def _log(series):
        return np.log(series + .000001)

    def meta(self, var, series):
        varname = "meta|" + self._rename_var(var)
        self._add_series(varname, series)
        self.metas.append(varname)

    def subject(self, var, series):
        varname = "subj|" + self._rename_var(var)
        self._add_series(varname, series)
        self.subjects.append(varname)

    def dependent(self, var, series):
        varname = "depvar|" + self._rename_var(var, VarType.Log)
        _series = self._log(series)
        self._add_series(varname, _series)
        self.depvar = varname

    def variable(self, var, series, vartype, bucket=None, detail=None):
        varname = "var|" + self._rename_var(var, vartype)
        if vartype == VarType.Log:
            self._add_series(varname, self._log(series))
            self.vars[varname] = (vartype, bucket, detail)
        elif vartype == VarType.Class:
            pfx = varname + "|"
            _d = pd.get_dummies(series, pfx)
            for k in _d:
                self._add_series(k, _d[k])
                self.vars[k] = (vartype, bucket, detail)
        else:
            self._add_series(varname, series)
            self.vars[varname] = (vartype, bucket, detail)

    def center(self):
        self.model_df = self.ds
        vars = list()
        vars.append(self.depvar)
        for _ in self.vars.keys():
            vars.append(_)

        aggkeys = dict()
        for _ in vars:
            aggkeys["mean_"+_] = (_, "mean")

        if self.subjects:
            self.means = self.model_df.groupby(self.subjects).agg(**aggkeys)
            self.model_df = self.model_df.merge(self.means, on=self.subjects)
        else:
            self.model_df["_zz"] = 1
            self.means = self.model_df.groupby("_zz").agg(**aggkeys)
            self.model_df = self.model_df.merge(self.means, on="_zz")

        self.means = self.means.reset_index()

        for _ in vars:
            self.model_df[_] = self.model_df[_] - self.model_df["mean_"+_]


    def run(self):
        self.center()

        X = self.model_df[self.vars.keys()]
        Y = self.model_df[self.depvar]
        model = sm.OLS(Y, X)
        result = model.fit()


        self.results["rsq"] = result.rsquared
        self.results["summary"] = result.summary()
        self.results["coefs"] = result.params.to_dict()
        self.results["mse"] = result.mse_resid
        self.results["resid"] = result.resid
        self.results["pred"] = result.fittedvalues
        self.results["full"] = result

    def decomp(self):
        self.decomp_df["ACTUAL"] = np.exp(self.ds[self.depvar])
        self.decomp_df["mult_resid"] = np.exp(self.results["resid"])

        all_vars = list(self.results["coefs"].keys())
        all_vars.append("resid")

        for k, v in self.results["coefs"].items():
            if self.vars[k][0] == VarType.Log:
                # Val is exp(val + Mean)
                # Ref is the exp(Mean)
                val = np.exp(self.model_df[k] + self.model_df["mean_" + k])
                ref = np.exp(self.model_df["mean_" + k])
                self.decomp_df["mult_" + k] = (val / ref)**(v)
            else:
                # Val is val + Mean
                # Ref is 0
                val = self.model_df[k] + self.model_df["mean_" + k]
                self.decomp_df["mult_" + k] = np.exp(val * v)

        small_base = list()
        pos_vols = list()
        neg_vols = list()
        var_vols = dict()

        for v in all_vars:
            var_vols[v] = list()

        for row in self.decomp_df.to_dict(orient="records"):
            mult_pos = 1
            mult_neg = 1

            pos_var = dict()
            neg_var = dict()

            for var in all_vars:
                m = row["mult_" + var]
                if m > 1:
                    mult_pos *= m
                    pos_var["mult_" + var] = m - 1
                elif m < 1:
                    mult_neg *= m
                    neg_var["mult_" + var] = m - 1

            mult_total = mult_pos * mult_neg
            smallb = row["ACTUAL"] / mult_total

            small_base.append(smallb)


            syn_a = mult_pos - 1
            syn_b = -1 * (mult_neg - 1)
            syn_ab = syn_a * syn_b

            pos_vol = smallb * syn_a
            neg_vol = smallb * -1 * (syn_b + syn_ab)

            pos_vols.append(pos_vol)
            neg_vols.append(neg_vol)

            psum = sum(pos_var.values())
            nsum = sum(neg_var.values())

            for v in all_vars:
                name = "mult_" + v
                if name in pos_var:
                    _v = pos_var[name] / psum * pos_vol
                    var_vols[v].append(_v)
                elif name in neg_var:
                    _v = neg_var[name] / nsum * neg_vol
                    var_vols[v].append(_v)
                else:
                    var_vols[v].append(0)

        self.decomp_df["small_base"] = np.array(small_base)
        self.decomp_df["pos_vols"] = np.array(pos_vols)
        self.decomp_df["neg_vols"] = np.array(neg_vols)

        for v in var_vols:
            self.decomp_df["decvol_"+v] = np.array(var_vols[v])

