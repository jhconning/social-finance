# socialfinance.py  -- for understanding contracts and bank funding structures
#

import numpy as np
import matplotlib.pyplot as plt


class Bank(object):
    ''' Lending Contracts, monitoring intensity and Bank funding structure in a 'neighborhood' or 'zone'
     where all borrowers have pledgeable assets A
    '''

    def __init__(self, A, beta):  # constructor to set initial default parameters.
        self.A = A  # pledgeable assets of borrowers
        self.gamma = 1.0  # cost of uninformed capital (1 + interest rate)
        self.beta = beta   #
        self.B0 = 30
        self.alpha = 0.4
        self.X = 200      # project return under success
        self.I = 100      # lump-sum investment
        self.p = 0.97     # prob. of success under diligence
        self.q = 0.80     # prob. of success under non-diligence
        self.F = 20        # Fixed cost per neighborhood
        self.K = 12000    # Intermediary capital in
        self.M = self.minmon(A, beta)
        self.Amax = 120

    def B(self, m):
        return self.B0 - self.alpha * m

    def FC(N):  # Avg fixed cost per borrower if bank has N borrowers
        return FO / N + f

    def AMe(self, m, beta):
        '''Minimum collateral for non-leveraged or equity-only MFI '''
        p, q, al, I, X = self.p, self.q, self.alpha, self.I, self.X    # for simpler formulas
        return  p/(p-q) * self.B(m)  - p * X + beta * I + m + beta*self.F

    def AM(self, m, beta):
        '''Minimum collateral for leveraged MFI '''
        p, q, al, I, X, gam = self.p, self.q, self.alpha, self.I, self.X, self.gamma    # for simpler formulas
        return p/(p-q) * self.B(m) - p * X + gam * I + m \
                  + ((beta - gam) / beta) * (q * m / (p - q)) + gam*self.F

    def Abest(self, m, beta):
        return np.minimum(self.AMe(m, beta), self.AM(m, beta))

    def Im(self, m):
        '''Minimum required equity investment by monitor'''
        return self.q * m / (self.p - self.q)

    def mcross(self, beta):
        '''Monitoring level at which equity only AMe and levered AM lines cross'''
        return beta * (self.I +self.F)* (self.p - self.q) / self.q

    def Across(self, beta):
        return self.AM(self.mcross(beta), beta)

    def mmax(self, beta):
        '''Maximal monitoring @ which equity-only monitor can break even'''
        return self.p * self.X - beta * (self.I + self.F)

    def Amin(self, beta):
        '''Lowest possible collateral requirement -- at max feasible monitoring'''
        return self.AMe(self.mmax(beta), beta)

    def mon(self, A, beta):
        '''optimal monitoring in leveraged MFI
           Zero if >A(0)'''
        AHI = self.AM(0, self.gamma)
        return (AHI - A) * (self.beta * (self.p - self.q)) / ((self.alpha - 1) * beta * self.p + self.gamma * self.q)

    def monE(self, A, beta):
        '''optimal monitoring in equity-only MFI'''
        AHI = self.AMe(0, beta)
        return (AHI - A) * ((self.p - self.q) / (self.q - (1 - self.alpha) * self.p))

    def minmon(self, A, beta):
        return np.minimum(self.monE(A, beta), self.mon(A, beta))

    def breturn(self,A, beta):
        '''borrower an array of borrower returns'''
        X, p,q, I, F, gam,  = self.X, self.p, self.q, self.I, self.F, self.gamma
        br = np.zeros(len(A))
        for i, a in enumerate(A):
            if a > self.AM(0, beta):
                br[i] = p * X - gam * I - gam * F
            elif (a <= self.AM(0, beta)) and (a > self.Across(beta)):
                br[i] = p * X - gam * I - gam * F - self.mon(a, beta) \
                    * (1 + ((beta - gam) / beta) * (q / (p - q)))
            elif (a <= self.Across(beta)) and (a >= self.Amin(beta)):
                br[i] = p * X - beta * I - beta * F - self.monE(a, beta)
            else:
                br[i] = 0
        return br

    def nreach(self,A, beta):
        '''number of borrowers reached with K of intermediary capital'''
        K,F, I = self.K, self.F, self.I

        nr = np.zeros(len(A))
        for i, a in enumerate(A):
            if a > self.AM(0, beta):
                nr[i] = float('NaN')
            elif (a <= self.AM(0, beta)) and (a > self.Across(beta)):
                nr[i] = K/(self.Im(self.mon(a,beta)) +F)
            elif (a <= self.Across(beta)) and (a >= self.Amin(beta)):
                nr[i] = K/(I+F)
            else:
                nr[i] = 0
        return nr

    def plotA(self, beta):
        mc = self.mcross(beta)
        Amc = self.AM(mc,beta)
        mx = self.mmax(beta)
        Amx = self.AMe(mx,beta)
        mm = np.linspace(0, self.Amax)
        mm_ = np.linspace(0, mx)
        fig, ax = plt.subplots(1)
        ax.plot(mm, self.AMe(mm, beta), label='equity only MFI',linestyle=':')
        ax.plot(mm,self.AM(mm,beta), label='leveraged MFI', linestyle=':')
        ax.plot(mm_,self.Abest(mm_, beta),linewidth=3.3)
        ax.set_xlim(0,self.mmax(beta)+10), ax.set_ylim(0,self.Amax)
        ax.set_title('Minimum Collateral requirement')
        ax.set_xlabel('monitoring intensity $m$')
        ax.set_ylabel('asset $A (m)$')
        ax.text(0,(Amx+Amc)/2, 'Equity-only', rotation='vertical',verticalalignment='center')
        ax.text(0, (Amc+self.AM(0,beta))/2, 'Debt', rotation='vertical',verticalalignment='center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.vlines(mc, ymin=0, ymax=Amc, linestyle =':')
        ax.hlines(Amc, xmin=0, xmax=mc, linestyle =':')
        ax.vlines(mx, ymin=0, ymax=Amx, linestyle =':')
        ax.hlines(Amx, xmin=0, xmax=mx, linestyle =':')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plotIm(self, beta):
        I, F = self.I, self.F
        mc = self.mcross(beta)
        Amc = self.AM(mc, beta)
        mx = self.mmax(beta)
        Amx = self.AMe(mx, beta)
        amin = self.Amin(beta)
        A_ = np.linspace(amin, self.AM(0, beta), self.Amax+1)  # color only loans
        Im = np.minimum(self.I + self.F, self.minmon(A_, beta) * (self.q / (self.p - self.q)) * (1 / beta))

        plt.title('Required monitoring m and investment Im')
        plt.plot(A_, Im, label=r'$I^m$ - monitoring equity')
        plt.plot(A_, I+F - Im, label=r'$I^u$ - uninformed debt')
        plt.xlabel('A -- pledgeable assets')
        plt.text(amin - 5, I+F, r'$I+F$')
        plt.text(amin + 2, I+F, r'$I^m$')
        plt.text(amin + 2, self.monE(amin, beta), 'm(A)')
        plt.text(amin + 2, 2, r'$I^u =I+F-I^m$')
        plt.plot(A_, self.minmon(A_, beta), label=r'$m$ - monitoring')
        plt.axvline(x= self.Amin(beta), linestyle=':')
        plt.axvline(x=Amc, linestyle=':')
        plt.axhline(y=I + F, linestyle=':')
        plt.axvline(x=self.AM(0, beta), linestyle=':')
        plt.ylim(0, I + F + 10)
        plt.xlim(self.Amin(beta) - 5, 100);

    def plotDE(self,beta):
        amin = self.Amin(beta)
        A_ = np.linspace(amin, self.AM(0, beta), self.Amax + 1)  # color only loans
        p,q, I, F = self.p, self.q, self.I, self.F
        plt.title('Debt to equity ratio:  ' + r'$\frac{I+F-I^m}{I^m}$')
        Im = np.minimum(I + F, self.minmon(A_, beta) * (q / (p - q)) * (1 / beta))
        de = (I + F - Im) / Im
        plt.plot(A_, de)
        plt.axvline(x=self.AM(self.mcross(beta), beta), linestyle=':');
        plt.axhline(y=0, linestyle=':');
        plt.axvline(x=self.Amin(beta), linestyle=':');

    def print_params(self):
        """
        Display scalar parameters alphabetically
        """
        params = sorted(vars(self).items())
        for itm in params:
            if np.isscalar(itm[1]):
                print(itm[0], '=', itm[1], end=', ')


if __name__ == '__main__':
    """Sample use of the bankzone class """



