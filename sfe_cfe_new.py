import numpy as np
import pandas as pd

def student_course_fixed_effect(sr, sc, tol=0.01):

    #remove UNIT_ERND_NBR = 0 and GRD_PNTS_PER_UNIT_NBR not NA or NaN
    sc = sc.loc[sc['UNITS_ERND_NBR'] > 0]
    sc = sc.loc[sc['GRD_PNTS_PER_UNIT_NBR'].notna()]
    sc = sc.loc[sc['GRD_PNTS_PER_UNIT_NBR'].notnull()]

    sc = sc.loc[sc['GRD_PNTS_PER_UNIT_NBR'] < 4.1]
    sc = sc.loc[sc['EARN_CR_IND'] > 0]

    #count the number of records in each TERM_CD,CLASS_NBR and remove records with TERM_CD,CLASS_NBR that has less than 2 records
    sc['count'] = sc.groupby(['TERM_CD','CLASS_NBR'])['CLASS_NBR'].transform('count')
    #sc = sc.loc[sc['count'] > 3]

    #also remove records with TERM_SHORT_DES that contains 'S' or 'M'
    sc = sc.loc[~sc['TERM_SHORT_DES'].str.contains('S|M')]

    TERML = sc.groupby('TERM_CD')['TERM_CD'].unique().sort_values()
    NTERM = len(TERML)
    sc['k_old'] = np.zeros(len(sc))
    sc['a_old'] = np.zeros(len(sc))

    #intialize SFE and CFE in sc
    sc['SFE'] = np.zeros(len(sc))
    sc['CFE'] = np.zeros(len(sc))
    sc['STDMEAN'] = np.zeros(len(sc))
    sc['CMEAN'] = np.zeros(len(sc))
    sc['ADJGRD'] = np.zeros(len(sc))
    sc['EFFORT'] = np.zeros(len(sc))

    for i in range(1):
        TERM_CD_test = int(TERML.iloc[i])
        print(TERM_CD_test)
        e = sc['TERM_CD'] <= TERM_CD_test
        e1 = sc.loc[e]['TERM_CD'] == TERM_CD_test
        kt = sc.loc[e].copy()
        #kt['k'] = kt.groupby('STDNT_ID').transform(lambda g: np.average(g['GRD_PNTS_PER_UNIT_NBR'], weights=g['UNITS_ERND_NBR']))
        #kt['a'] = kt.groupby('CLASS_NBR').transform(lambda g: np.average(g['GRD_PNTS_PER_UNIT_NBR'], weights=g['UNITS_ERND_NBR']))

        res = kt.groupby('STDNT_ID').apply(lambda g: np.average(kt.loc[g.index, 'GRD_PNTS_PER_UNIT_NBR'],
                                                                weights=kt.loc[g.index,'UNITS_ERND_NBR']))
        kt['k'] = kt['STDNT_ID'].map(res)

        res = kt.groupby('CLASS_NBR').apply(lambda g: np.average(kt.loc[g.index, 'GRD_PNTS_PER_UNIT_NBR'],
                                                                weights=kt.loc[g.index, 'UNITS_ERND_NBR']))
        kt['a'] = kt['CLASS_NBR'].map(res)

        #kt['a'] = kt.groupby('CLASS_NBR').apply(
        #    lambda g: (
        #        print("GRD_PNTS_PER_UNIT_NBR:", g['GRD_PNTS_PER_UNIT_NBR']),
        #        print("a:", g['a']),
        #        print("UNITS_ERND_NBR:", g['UNITS_ERND_NBR']),
        #        print(g['CLASS_NBR']),
        #        input("Press Enter to continue..."),
        #        np.average(g['GRD_PNTS_PER_UNIT_NBR'] - g['a'], weights=g['UNITS_ERND_NBR'])
        #    )[3]
        #).reset_index(level=0, drop=True)

        #kt['a'] = kt.groupby('CLASS_NBR').apply(
        #        lambda g: (
        #            print("GRD_PNTS_PER_UNIT_NBR:", g['GRD_PNTS_PER_UNIT_NBR']),
        #            print("UNITS_ERND_NBR:", g['UNITS_ERND_NBR']),
        #            print(g['CLASS_NBR']),
        #            print(np.average(g['GRD_PNTS_PER_UNIT_NBR'], weights=g['UNITS_ERND_NBR'])),
        #            input("Press Enter to continue...")
        #        )[3]
        #     ).values

        #print kt['a'] values that are not nan
        print(kt['a'].loc[~np.isnan(kt['a'])])
        print(kt['k'].loc[~np.isnan(kt['k'])])

        kt['CMEAN'] = kt['a']
        kt['STDMEAN'] = kt['k']
        kt['ADJGRD'] = kt['GRD_PNTS_PER_UNIT_NBR'] - kt['CMEAN']
        kt['k_old'] = kt['k']
        kt['a_old'] = kt['a']
        a_diff_max = 100
        k_diff_max = 100

        while (a_diff_max > tol) or (k_diff_max > tol):
            #kt['k'] = kt.groupby('STDNT_ID').apply(
            #    lambda g: np.average(g['GRD_PNTS_PER_UNIT_NBR'] - g['a'], weights=g['UNITS_ERND_NBR'])).reset_index(level=0,drop=True)
            #kt['a'] = kt.groupby('CLASS_NBR').apply(
            #    lambda g: np.average(g['GRD_PNTS_PER_UNIT_NBR'] - g['k'], weights=g['UNITS_ERND_NBR'])).reset_index(level=0,drop=True)

            res = kt.groupby('STDNT_ID').apply(lambda g: np.average(kt.loc[g.index, 'GRD_PNTS_PER_UNIT_NBR'],
                                                                    weights=kt.loc[g.index, 'UNITS_ERND_NBR']))
            kt['k'] = kt['STDNT_ID'].map(res)

            res = kt.groupby('CLASS_NBR').apply(lambda g: np.average(kt.loc[g.index, 'GRD_PNTS_PER_UNIT_NBR'],
                                                                     weights=kt.loc[g.index, 'UNITS_ERND_NBR']))
            kt['a'] = kt['CLASS_NBR'].map(res)

            kdiff = abs(kt['k'] - kt['k_old'])
            adiff = abs(kt['a'] - kt['a_old'])

            #print kt['k'] values that are not nan
            print(kt['k'].loc[~np.isnan(kt['k'])])
            #print(kt['k_old'].loc[~np.isnan(kt['k_old'])])
            print(kt['a'].loc[~np.isnan(kt['a'])])

            kt['k_old'] = kt['k']
            kt['a_old'] = kt['a']
            ss = np.nansum((kt['GRD_PNTS_PER_UNIT_NBR'] - kt['k'] - kt['a'])**2)
            a_diff_max = np.nanmax(kdiff)
            k_diff_max = np.nanmax(adiff)

        sc.loc[e, 'CMEAN'][e1] = kt.loc[e1,'CMEAN']
        sc.loc[e, 'STDMEAN'][e1] = kt.loc[e1,'STDMEAN']
        sc.loc[e, 'CFE'][e1] = kt.loc[e1,'a']
        sc.loc[e, 'SFE'][e1] = kt.loc[e1,'k']
        # sc.loc[e, 'EFFORT'][e1] = (kt.loc[e1,'GRD_PNTS_PER_UNIT_NBR'] - kt.loc[e1,'SFE'] - kt.loc[e1,'CFE']) * kt.loc[e1'UNITS_ERND_NBR']

    sr = sr.merge(sc.groupby('STDNT_ID').agg({"SFE": "mean", "CFE": "mean", "EFFORT": "sum"}), on="STDNT_ID")
    return sc