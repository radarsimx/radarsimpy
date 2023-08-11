function Sout = add_phase_noise( Sin, Fs, phase_noise_freq, phase_noise_power, VALIDATION_ON )
%
% function Sout = add_phase_noise( Sin, Fs, phase_noise_freq, phase_noise_power, VALIDATION_ON )
%
% Oscillator Phase Noise Model
% 
%  INPUT:
%     Sin - input COMPLEX signal
%     Fs  - sampling frequency ( in Hz ) of Sin
%     phase_noise_freq  - frequencies at which SSB Phase Noise is defined (offset from carrier in Hz)
%     phase_noise_power - SSB Phase Noise power ( in dBc/Hz )
%     VALIDATION_ON  - 1 - perform validation, 0 - don't perfrom validation
%
%  OUTPUT:
%     Sout - output COMPLEX phase noised signal
%
%  NOTE:
%     Input signal should be complex
%
%  EXAMPLE ( How to use add_phase_noise ):
%         Assume SSB Phase Noise is specified as follows:
%      -------------------------------------------------------
%      |  Offset From Carrier      |        Phase Noise      |
%      -------------------------------------------------------
%      |        1   kHz            |        -84  dBc/Hz      |
%      |        10  kHz            |        -100 dBc/Hz      |
%      |        100 kHz            |        -96  dBc/Hz      |
%      |        1   MHz            |        -109 dBc/Hz      |
%      |        10  MHz            |        -122 dBc/Hz      |
%      -------------------------------------------------------
%
%      Assume that we have 10000 samples of complex sinusoid of frequency 3 KHz 
%      sampled at frequency 40MHz:
%       
%       Fc = 3e3; % carrier frequency
%       Fs = 40e6; % sampling frequency
%       t = 0:9999;
%       S = exp(j*2*pi*Fc/Fs*t); % complex sinusoid
%
%      Then, to produce phase noised signal S1 from the original signal S run follows:
%
%       Fs = 40e6;
%       phase_noise_freq = [ 1e3, 10e3, 100e3, 1e6, 10e6 ]; % Offset From Carrier
%       phase_noise_power = [ -84, -100, -96, -109, -122 ]; % Phase Noise power
%       S1 = add_phase_noise( S, Fs, phase_noise_freq, phase_noise_power );

% Version 1.0
% Alex Bur-Guy, October 2005
% alex@wavion.co.il
%
% Revisions:
%       Version 1.5 -   Comments. Validation.
%       Version 1.0 -   initial version

% NOTES:
% 1)  The presented model is a simple VCO phase noise model based on the following consideration:
% If the output of an oscillator is given as  V(t) = V0 * cos( w0*t + phi(t) ), 
% then phi(t)  is defined as the phase noise.  In cases of small noise
% sources (a valid assumption in any usable system), a narrowband modulation approximation can
% be used to express the oscillator output as:
% 
% V(t) = V0 * cos( w0*t + phi(t) )
% 
%        = V0 * [cos(w0*t)*cos(phi(t)) - sin(w0*t)*sin(phi(t)) ]
% 
%        ~ V0 * [cos(w0*t) - sin(w0*t)*phi(t)] 
% 
% This shows that phase noise will be mixed with the carrier to produce sidebands around the carrier.
%
% 
% 2) In other words, exp(j*x) ~ (1+j*x) for small x
%
% 3) Phase noise = 0 dBc/Hz at freq. offset of 0 Hz
% 
% 4) The lowest phase noise level is defined by the input SSB phase noise power at the maximal 
%    freq. offset from DC. (IT DOES NOT BECOME EQUAL TO ZERO )
% 
% The generation process is as follows:
%  First of all we interpolate (in log-scale) SSB phase noise power spectrum in M 
%  equally spaced points (on the interval [0 Fs/2] including bounds ).
%
%  After that we calculate required frequency shape of the phase noise by X(m) = sqrt(P(m)*dF(m)) 
%  and after that complement it by the symmetrical negative part of the spectrum.
%
%  After that we generate AWGN of power 1 in the freq domain and multiply it sample-by-sample to 
%  the calculated shape 
%
%  Finally we perform  2*M-2 points IFFT to such generated noise
%  ( See comments inside the code )
% 
%  0 dBc/Hz                                
%  \                                                          /
%   \                                                        / 
%    \                                                      /  
%     \P dBc/Hz                                            /   
%     .\                                                  /    
%     . \                                                /     
%     .  \                                              /      
%     .   \____________________________________________/  /_ This level is defined by the phase_noise_power at the maximal freq. offset from DC defined in phase_noise_freq   
%     .                                                   \        
%  |__| _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__   (N points)
%  0   dF                       Fs/2                          Fs
%  DC
%
%
%  For some basics about Oscillator phase noise see:
%     http://www.circuitsage.com/pll/plldynamics.pdf
%
%     http://www.wj.com/pdf/technotes/LO_phase_noise.pdf

if nargin < 5
     VALIDATION_ON = 0;
end

% Check Input
error( nargchk(4,5,nargin) );

if ~any( imag(Sin(:)) )
     error( 'Input signal should be complex signal' );
end
if max(phase_noise_freq) >= Fs/2
     error( 'Maximal frequency offset should be less than Fs/2');
end
     
% Make sure phase_noise_freq and  phase_noise_power are the row vectors
phase_noise_freq = phase_noise_freq(:).';
phase_noise_power = phase_noise_power(:).';
if length( phase_noise_freq ) ~= length( phase_noise_power )
     error('phase_noise_freq and phase_noise_power should be of the same length');
end

% Sort phase_noise_freq and phase_noise_power
[phase_noise_freq, indx] = sort( phase_noise_freq );
phase_noise_power = phase_noise_power( indx );

% Add 0 dBc/Hz @ DC
if ~any(phase_noise_freq == 0)
     phase_noise_power = [ 0, phase_noise_power ];
     phase_noise_freq = [0, phase_noise_freq];
end

% Calculate input length
N = prod( size( Sin ) );

% Define M number of points (frequency resolution) in the positive spectrum 
%  (M equally spaced points on the interval [0 Fs/2] including bounds), 
% then the number of points in the negative spectrum will be M-2 
%  ( interval (Fs/2, Fs) not including bounds )
%
% The total number of points in the frequency domain will be 2*M-2, and if we want 
%  to get the same length as the input signal, then
%   2*M-2 = N
%   M-1 = N/2
%   M = N/2 + 1
%
%  So, if N is even then M = N/2 + 1, and if N is odd we will take  M = (N+1)/2 + 1
%
if rem(N,2),    % N odd
     M = (N+1)/2 + 1;
else
     M = N/2 + 1;
end


% Equally spaced partitioning of the half spectrum
F  = linspace( 0, Fs/2, M );    % Freq. Grid 
dF = [diff(F) F(end)-F(end-1)]; % Delta F


% Perform interpolation of phase_noise_power in log-scale
intrvlNum = length( phase_noise_freq );
logP = zeros( 1, M );
for intrvlIndex = 1 : intrvlNum,
     leftBound = phase_noise_freq(intrvlIndex);
     t1 = phase_noise_power(intrvlIndex);
     if intrvlIndex == intrvlNum
          rightBound = Fs/2; 
          t2 = phase_noise_power(end);
          inside = find( F>=leftBound & F<=rightBound );  
     else
          rightBound = phase_noise_freq(intrvlIndex+1); 
          t2 = phase_noise_power(intrvlIndex+1);
          inside = find( F>=leftBound & F<rightBound );
     end
     logP( inside ) = ...
          t1 + ( log10( F(inside) + realmin) - log10(leftBound+ realmin) ) / ( log10( rightBound + realmin) - log10( leftBound + realmin) ) * (t2-t1);     
end
P = 10.^(real(logP)/10); % Interpolated P ( half spectrum [0 Fs/2] ) [ dBc/Hz ]

% Now we will generate AWGN of power 1 in frequency domain and shape it by the desired shape
% as follows:
%
%    At the frequency offset F(m) from DC we want to get power Ptag(m) such that P(m) = Ptag/dF(m),
%     that is we have to choose X(m) =  sqrt( P(m)*dF(m) );
%  
% Due to the normalization factors of FFT and IFFT defined as follows:
%     For length K input vector x, the DFT is a length K vector X,
%     with elements
%                      K
%        X(k) =       sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
%                     n=1
%     The inverse DFT (computed by IFFT) is given by
%                      K
%        x(n) = (1/K) sum  X(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
%                     k=1
%
% we have to compensate normalization factor (1/K) multiplying X(k) by K.
% In our case K = 2*M-2.

% Generate AWGN of power 1

if ~VALIDATION_ON
     awgn_P1 = ( sqrt(0.5)*(randn(1, M) +1j*randn(1, M)) );
else
     awgn_P1 = ( sqrt(0.5)*(ones(1, M) +1j*ones(1, M)) );
end

% Shape the noise on the positive spectrum [0, Fs/2] including bounds ( M points )
X = (2*M-2) * sqrt( dF .* P ) .* awgn_P1; 

% Complete symmetrical negative spectrum  (Fs/2, Fs) not including bounds (M-2 points)
X( M + (1:M-2) ) = fliplr( conj(X(2:end-1)) ); 

% Remove DC
X(1) = 0; 

% Perform IFFT 
x = ifft( X ); 

% Calculate phase noise 
phase_noise = exp( j * real(x(1:N)) );

% Add phase noise
if ~VALIDATION_ON
     Sout = Sin .* reshape( phase_noise, size(Sin) );
else
     Sout = 'VALIDATION IS ON';
end

if VALIDATION_ON
     figure; 
     plot( phase_noise_freq, phase_noise_power, 'o-' ); % Input SSB phase noise power
     hold on;    
     grid on;     
     plot( F, 10*log10(P),'r*-'); % Input SSB phase noise power
     X1 = fft( phase_noise );
     plot( F, 10*log10( ( (abs(X1(1:M))/max(abs(X1(1:M)))).^2 ) ./ dF(1) ), 'ks-' );% generated phase noise exp(j*x)     
     X2 = fft( 1 + j*real(x(1:N)) ); 
     plot( F, 10*log10( ( (abs(X2(1:M))/max(abs(X2(1:M)))).^2 ) ./ dF(1) ), 'm>-' ); % approximation ( 1+j*x )   
     xlabel('Frequency [Hz]');
     ylabel('dBc/Hz');
     legend( ...
          'Input SSB phase noise power', ...
          'Interpolated SSB phase noise power', ...
          'Positive spectrum of the generated phase noise exp(j*x)', ...
          'Positive spectrum of the approximation ( 1+j*x )' ...
     );     
end
