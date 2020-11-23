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
        self.original_df = pd.DataFrame()
        self.n_cols = 0
        self.meta_vars = list()
        self.subject_vars = list()
        self.dependent_var = None
        self.model_vars = dict()

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
        self._wide_decomp_df = pd.DataFrame()

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
        self.original_df.insert(self.n_cols, var, series)
        self.n_cols += 1

    @staticmethod
    def _log(series):
        return np.log(series + .000001)

    def meta(self, var, series):
        varname = "meta|" + self._rename_var(var)
        self._add_series(varname, series)
        self.meta_vars.append(varname)

    def subject(self, var, series):
        varname = "subj|" + self._rename_var(var)
        self._add_series(varname, series)
        self.subject_vars.append(varname)

    def dependent(self, var, series):
        varname = "depvar|" + self._rename_var(var, VarType.Log)
        _series = self._log(series)
        self._add_series(varname, _series)
        self.dependent_var = varname

    def variable(self, var, series, vartype, bucket=None, detail=None):
        varname = "var|" + self._rename_var(var, vartype)
        if vartype == VarType.Log:
            self._add_series(varname, self._log(series))
            self.model_vars[varname] = (vartype, bucket, detail)
        elif vartype == VarType.Class:
            pfx = varname + "|"
            _d = pd.get_dummies(series, pfx)
            for k in _d:
                self._add_series(k, _d[k])
                self.model_vars[k] = (vartype, bucket, detail)
        else:
            self._add_series(varname, series)
            self.model_vars[varname] = (vartype, bucket, detail)

    def center(self):
        self.model_df = self.original_df
        vars = list()
        vars.append(self.dependent_var)
        for _ in self.model_vars.keys():
            vars.append(_)

        aggkeys = dict()
        for _ in vars:
            aggkeys["mean_"+_] = (_, "mean")

        if self.subject_vars:
            self.means = self.model_df.groupby(self.subject_vars).agg(**aggkeys)
            self.model_df = self.model_df.merge(self.means, on=self.subject_vars)
        else:
            self.model_df["_zz"] = 1
            self.means = self.model_df.groupby("_zz").agg(**aggkeys)
            self.model_df = self.model_df.merge(self.means, on="_zz")

        self.means = self.means.reset_index()

        for _ in vars:
            self.model_df[_] = self.model_df[_] - self.model_df["mean_"+_]


    def run(self):
        self.center()

        X = self.model_df[list(self.model_vars.keys())]
        Y = self.model_df[self.dependent_var]
        model = sm.OLS(Y, X)
        result = model.fit()


        self.results["rsq"] = result.rsquared
        self.results["summary"] = result.summary()
        self.results["coefs"] = result.params.to_dict()
        self.results["mse"] = result.mse_resid
        self.results["resid"] = result.resid
        self.results["pred"] = result.fittedvalues
        self.results["full"] = result

        self.decomp()

    def decomp(self):
        _wide_decomp = pd.DataFrame()
        for _ in self.meta_vars:
            _wide_decomp[_] = self.original_df[_]

        for _ in self.subject_vars:
            _wide_decomp[_] = self.original_df[_]

        _wide_decomp["ACTUAL"] = np.exp(self.original_df[self.dependent_var])
        _wide_decomp["APE"] = np.abs(np.exp(self.results["resid"]) - 1)
        _wide_decomp["mult_resid"] = np.exp(self.results["resid"])

        dec_vars = list(self.results["coefs"].keys())
        dec_vars.append("resid")

        for k, v in self.results["coefs"].items():
            if self.model_vars[k][0] == VarType.Log:
                # Val is exp(val + Mean)
                # Ref is the exp(Mean)
                val = np.exp(self.model_df[k] + self.model_df["mean_" + k])
                ref = np.exp(self.model_df["mean_" + k])
                _wide_decomp["mult_" + k] = (val / ref)**(v)
            else:
                # Val is val + Mean
                # Ref is 0
                val = self.model_df[k] + self.model_df["mean_" + k]
                _wide_decomp["mult_" + k] = np.exp(val * v)

        nc_reference_base_vol = list()
        nc_posvol = list()
        nc_negvol = list()
        nc_dec_var_vol = dict()

        for v in dec_vars:
            nc_dec_var_vol[v] = list()

        for row in _wide_decomp.to_dict(orient="records"):
            positive_multipliers = 1
            negative_multipliers = 1

            positive_vars_magnitude = dict()
            negative_vars_magnitude = dict()

            for var in dec_vars:
                m = row["mult_" + var]
                if m > 1:
                    positive_multipliers *= m
                    positive_vars_magnitude["mult_" + var] = m - 1
                elif m < 1:
                    negative_multipliers *= m
                    negative_vars_magnitude["mult_" + var] = m - 1

            total_multiplier = positive_multipliers * negative_multipliers
            reference_base_vol = row["ACTUAL"] / total_multiplier

            nc_reference_base_vol.append(reference_base_vol)


            synergy_pos_comp = positive_multipliers - 1
            synergy_neg_comp = -1 * (negative_multipliers - 1)
            synergy_comb_comp = synergy_pos_comp * synergy_neg_comp

            pos_vol = reference_base_vol * synergy_pos_comp
            neg_vol = reference_base_vol * -1 * (synergy_neg_comp +
                                                 synergy_comb_comp)

            nc_posvol.append(pos_vol)
            nc_negvol.append(neg_vol)

            psum = sum(positive_vars_magnitude.values())
            nsum = sum(negative_vars_magnitude.values())

            for v in dec_vars:
                name = "mult_" + v
                if name in positive_vars_magnitude:
                    _v = positive_vars_magnitude[name] / psum * pos_vol
                    nc_dec_var_vol[v].append(_v)
                elif name in negative_vars_magnitude:
                    _v = negative_vars_magnitude[name] / nsum * neg_vol
                    nc_dec_var_vol[v].append(_v)
                else:
                    nc_dec_var_vol[v].append(0)

        _wide_decomp["decvol_small_base"] = np.array(nc_reference_base_vol)
        _wide_decomp["pos_vols"] = np.array(nc_posvol)
        _wide_decomp["neg_vols"] = np.array(nc_negvol)

        for v in nc_dec_var_vol:
            _wide_decomp["decvol_"+v] = np.array(nc_dec_var_vol[v])

        self._wide_decomp_df = _wide_decomp

        _long_obs = list()
        _long_cols = list()
        _long_cols.append("MODEL_NAME")

        for _ in self.meta_vars:
            _long_cols.append(_)

        for _ in self.subject_vars:
            _long_cols.append(_)

        _long_cols.append("BUCKET")
        _long_cols.append("DETAIL")
        _long_cols.append("VARIABLE")
        _long_cols.append("DECOMPOSED_VOLUME")

        dec_vars.append("small_base")

        for row in _wide_decomp.to_dict(orient="records"):

            for dc in dec_vars:
                record = list()
                record.append(self.name)
                for _ in self.meta_vars:
                    record.append(row[_])
                for _ in self.subject_vars:
                    record.append(row[_])
                if dc == "resid":
                    record.append("BASE")
                    record.append("UNEXPLAINED")
                elif dc == "small_base":
                    record.append("BASE")
                    record.append("REFERENCE")
                else:
                    record.append(self.model_vars[dc][1])
                    record.append(self.model_vars[dc][2])
                record.append(dc)
                record.append(row["decvol_"+dc])

                _long_obs.append(tuple(record))

        self.decomp_df = pd.DataFrame(_long_obs, columns=_long_cols)

        print(self.decomp_df)
