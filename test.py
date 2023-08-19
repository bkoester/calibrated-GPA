import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.optimize, scipy.stats
import awswrangler as wr

import warnings

# https://github.com/bkoester/calibrated-GPA/blob/main/Synth%20GPA%20v4%20linear.ipynb

def run_full_model():

    warnings.filterwarnings("ignore")
    
    np.set_printoptions(precision=18)
    
    bucket_path='s3://umich-larc-test/stage1/'
    file=bucket_path+'LARC_STDNT_CLASS_INFO'
    grades = wr.s3.read_parquet(file)
    
    #keep only rows with GRD_BASIS_ENRL_DES containing 'graded'
    grades = grades[grades['GRD_BASIS_ENRL_DES'].str.contains('Graded')]
    
    #and only keep GRD_PNTS_PER_UNITS_NBR <= 4.0 and CRSE_GRD_OFFCL_CD is no
    #this isn't perfect because it will include grad students
    grades = grades[grades['GRD_PNTS_PER_UNIT_NBR'] <= 4.0]
    #grades = grades[grades['TERM_CD'] > 2160]
    
    #remove rows with missing data in GRD_PNTS_PER_UNITS_NBR
    grades = grades[pd.notnull(grades['GRD_PNTS_PER_UNIT_NBR'])]
    
    #default_c_params = np.array([0.0, 1.0, 0.0])
    #def p(s, c):
    #    return c[:,2] + (4 - c[:,2])/(1 + np.exp(-c[:,1]*(s[:,0] - c[:,0])))
    default_c_params = np.array([2.0, 1.0])
    def p(s, c):
        return c[:,1] * s[:,0] + c[:,0]
    def s_penalty(s):
        return 1e-2 * s[0]**2
    #def c_penalty(c):
    #    return penalty_factor * (0.01 * c[0]**2 + (10 * (c[1] < 0) + (c[1] > 0)) * c[1]**2)
    def c_penalty(c):
        return 1e-4 * c[0]**2 + 1e-2 * c[1]**2
        
    grades['Course']=grades['SBJCT_CD']+grades['CATLG_NBR']
    grades['Score'] = grades['GRD_PNTS_PER_UNIT_NBR']
    
    #use their course size filters
    grades = grades.groupby('STDNT_ID').filter(lambda x: len(x) >= 12)
    grades = grades.groupby('Course').filter(lambda x: len(x) >= 30)
    
    
    #now repack for efficiency
    grades_by_student = grades.groupby('STDNT_ID')
    grades_by_course = grades.groupby('Course')
    student_ids = np.array(sorted(list(grades_by_student.groups.keys())))
    courses = np.array(sorted(list(grades_by_course.groups.keys())))
    
    n_students = len(student_ids)
    n_courses = len(courses)
    n_grades = grades.shape[0]
    
    student_idx_by_id = {}
    for (s_idx, id) in enumerate(student_ids):
        student_idx_by_id[id] = s_idx
    course_idx_by_course = {course: c_idx for (c_idx, course) in enumerate(courses)}
    course_idx_by_student_idx = {}
    true_scores_by_student_idx = {}
    for (s_idx, id) in enumerate(student_ids):
        course_idx_by_student_idx[s_idx] = np.array([course_idx_by_course[course] for course in grades_by_student.get_group(id)['Course']])
        true_scores_by_student_idx[s_idx] = grades_by_student.get_group(id)['Score'].values
    student_idx_by_course_idx = {}
    true_scores_by_course_idx = {}
    for (c_idx, course) in enumerate(courses):
        student_idx_by_course_idx[c_idx] = np.array([student_idx_by_id[id] for id in grades_by_course.get_group(course)['STDNT_ID']])
        true_scores_by_course_idx[c_idx] = grades_by_course.get_group(course)['Score'].values
    
    ###########This is just a plotting interlude#######
    #make a few plots first about student level GPAs
    s_gpa = np.zeros(n_students)
    s_gpa_std = np.zeros(n_students)
    for s_idx in range(n_students):
        true_scores = true_scores_by_student_idx[s_idx]
        s_gpa[s_idx] = true_scores.mean()
        s_gpa_std[s_idx] = true_scores.std()
    plt.hist(s_gpa, bins=np.linspace(0, 4, 41))
    plt.grid()
    plt.xlabel('raw student GPA')
    plt.ylabel('number of students');
    
    plt.hist2d(s_gpa, s_gpa_std, [np.linspace(0, 4, 100), np.linspace(0, 2, 80)], norm=matplotlib.colors.LogNorm());
    plt.grid()
    x = np.linspace(0, 4, 300)
    plt.plot(x, np.sqrt((x/4) * (1 - (x/4))) * 4, 'k--')
    print("Dashed line shows theoretical upper-bound")
    cbar = plt.colorbar()
    cbar.set_label('number of studnets')
    plt.xlabel('raw student GPA');
    plt.ylabel('raw student grade std dev');
    
    #and second about course level GPAs
    c_gpa = np.zeros(n_courses)
    c_gpa_std = np.zeros(n_courses)
    for c_idx in range(n_courses):
        true_scores = true_scores_by_course_idx[c_idx]
        c_gpa[c_idx] = true_scores.mean()
        c_gpa_std[c_idx] = true_scores.std()
    plt.hist(c_gpa, bins=np.linspace(0, 4, 41))
    plt.grid()
    plt.xlabel('raw course GPA')
    plt.ylabel('number of courses');
    
    plt.hist2d(c_gpa, c_gpa_std, [np.linspace(0, 4, 100), np.linspace(0, 2, 80)], norm=matplotlib.colors.LogNorm());
    plt.grid()
    x = np.linspace(0, 4, 300)
    plt.plot(x, np.sqrt((x/4) * (1 - (x/4))) * 4, 'k--')
    print("Dashed line shows theoretical upper-bound")
    cbar = plt.colorbar()
    cbar.set_label('number of courses')
    plt.xlabel('raw course GPA');
    plt.ylabel('raw course grade std dev');
    
    ######################
    
    def total_cost():
        V = 0
        for s_idx in range(n_students):
            loc_s = s[s_idx,:]
            loc_c = c[course_idx_by_student_idx[s_idx],:]
            true_scores = true_scores_by_student_idx[s_idx]
            pred_scores = p(np.atleast_2d(loc_s), loc_c)
            V += np.sum((true_scores - pred_scores)**2) / n_grades
        for s_idx in range(n_students):
            V += s_penalty(s[s_idx,:]) / n_students
        for c_idx in range(n_courses):
            V += c_penalty(c[c_idx,:]) / n_courses
        return V
    # update a single student parameter from all their courses
    def update_one_s(s_idx):
        init_loc_s = s[s_idx,:]
        loc_c = c[course_idx_by_student_idx[s_idx],:]
        true_scores = true_scores_by_student_idx[s_idx]
        def cost(loc_s):
            pred_scores = p(np.atleast_2d(loc_s), loc_c)
            v = np.sum((true_scores - pred_scores)**2) * n_students / n_grades
            return v + s_penalty(loc_s)
        res = scipy.optimize.minimize(cost, init_loc_s)
        return res.x
        
    def update_one_c(c_idx):
        init_loc_c = c[c_idx,:]
        loc_s = s[student_idx_by_course_idx[c_idx],:]
        true_scores = true_scores_by_course_idx[c_idx]
        def cost(loc_c):
            pred_scores = p(loc_s, np.atleast_2d(loc_c))
            v = np.sum((true_scores - pred_scores)**2) * n_courses / n_grades
            return v + c_penalty(loc_c)
        res = scipy.optimize.minimize(cost, init_loc_c)
        return res.x
    # update all student parameters given fixed course parameters
    def update_s():
        for s_idx in range(n_students):
            #if s_idx % 10000 == 0:
            #    print("%d/%d" % (s_idx, n_students))
            s[s_idx,:] = update_one_s(s_idx)
    # update all course parameters given fixed student parameters
    def update_c():
        for c_idx in range(n_courses):
            #if c_idx % 1000 == 0:
            #    print("%d/%d" % (c_idx, n_courses))
            c[c_idx,:] = update_one_c(c_idx)
    
    # find a single best student parameter
    def search_one_s(s_idx):
        loc_c = c[course_idx_by_student_idx[s_idx],:]
        true_scores = true_scores_by_student_idx[s_idx]
        def cost(loc_s):
            pred_scores = p(np.atleast_2d(loc_s), loc_c)
            v = np.linalg.norm((true_scores - pred_scores) / true_scores.shape[0])
            return v + s_penalty(loc_s)
        v_opt = 1e10
        loc_s_opt = 0
        for s0 in np.linspace(-5, 5, 10):
            loc_s = np.array([s0])
            v = cost(loc_s)
            if v < v_opt:
                v_opt = v
                loc_s_opt = loc_s
        return loc_s_opt
    # find a single best course parameter
    def search_one_c(c_idx):
        loc_s = s[student_idx_by_course_idx[c_idx],:]
        true_scores = true_scores_by_course_idx[c_idx]
        def cost(loc_c):
            pred_scores = p(loc_s, np.atleast_2d(loc_c))
            v = np.linalg.norm((true_scores - pred_scores) / true_scores.shape[0])
            return v + c_penalty(loc_c)
        v_opt = 1e10
        loc_c_opt = 0
        for c0 in np.linspace(-10, 0, 10):
            for c1 in np.linspace(0, 2.5, 10):
                loc_c = np.array([c0, c1])
                v = cost(loc_c)
                if v < v_opt:
                    v_opt = v
                    loc_c_opt = loc_c
        return loc_c_opt
        
    def search_s():
        #max_ds = 0
        for s_idx in range(n_students):
            #if s_idx % 1000 == 0:
                #print('%d/%d' % (s_idx, n_students))
            #    print('%d/%d: max_ds = %f' % (s_idx, n_students, max_ds))
            new_loc_s = search_one_s(s_idx)
            s[s_idx,:] = new_loc_s
            #ds = np.abs(new_loc_s[0] - s[s_idx,0])
            #max_ds = max(max_ds, ds)
        #return max_ds
    #max_ds = search_s()
    #print('max_ds = %f' % max_ds)
    def search_c():
        #max_dc = np.array([0, 0])
        for c_idx in range(n_courses):
            #if c_idx % 100 == 0:
                #print('%d/%d' % (c_idx, n_courses))
            #    print('%d/%d: max_dc = %s' % (c_idx, n_courses, max_dc))
            new_loc_c = search_one_c(c_idx)
            c[c_idx,:] = new_loc_c
            #dc = np.abs(new_loc_c - c[c_idx,:])
            #max_dc = np.maximum(max_dc, dc)
        #return max_dc
    #max_dc = search_c()
    #print('max_dc = %s' % max_dc)
    
    s = np.zeros((n_students, 1))
    s[:,0] = (s_gpa - s_gpa.mean()) / s_gpa.std()
    # initialize students with their GPA rank
    ranks = [(s_i, s_gpa[s_i]) for s_i in range(n_students)]
    ranks.sort(key = lambda r: r[1])
    s = np.zeros((n_students, 1))
    for (i, r) in enumerate(ranks):
        s[r[0]] = i
    s[:,0] = (s[:,0] - s[:,0].mean()) / s[:,0].std()
    # initialize courses with default values
    c = np.zeros((n_courses, len(default_c_params)))
    for c_idx in range(n_courses):
        c[c_idx,:] = default_c_params
        
    def iterate():
        max_iteration = 4
        for i in range(max_iteration):
            s_old = s.copy()
            c_old = c.copy()
            v_old = total_cost()
            search_c()
            search_s()
            ds = np.linalg.norm(s - s_old) / s.shape[0]
            dc = np.linalg.norm(c - c_old) / c.shape[0]
            v = total_cost()
            dv = np.abs(v - v_old)
            print("searth iteration %d, ds = %f, dc = %f, dv = %f" % (i, ds, dc, dv))
    
        max_iteration = 5
        for i in range(max_iteration):
            s_old = s.copy()
            c_old = c.copy()
            v_old = total_cost()
            update_c()
            update_s()
            ds = np.linalg.norm(s - s_old) / s.shape[0]
            dc = np.linalg.norm(c - c_old) / c.shape[0]
            v = total_cost()
            dv = np.abs(v - v_old)
            print("opt iteration %d, ds = %f, dc = %f, dv = %f" % (i, ds, dc, dv))
            if ds < 1e-6 and dc < 1e-6:
                print("successful termination")
                break
        else:
            print("iteration did not successfully terminate")
            
    iterate()
        
    print("s_0: [%f,%f], mu = %f, std = %f" % (s[:,0].min(), s[:,0].max(), s[:,0].mean(), s[:,0].std()))
    print("c_0: [%f,%f], mu = %f, std = %f" % (c[:,0].min(), c[:,0].max(), c[:,0].mean(), c[:,0].std()))
    print("c_1: [%f,%f], mu = %f, std = %f" % (c[:,1].min(), c[:,1].max(), c[:,1].mean(), c[:,1].std()))

    return s, c