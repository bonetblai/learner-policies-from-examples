% ==== Input ====

#defined feature/1.      % feature(F).      [F is feature]
#defined boolean/1.      % boolean(F).      [F is boolean feature]
#defined good/1.         % good(G).         [G is good transition (example)]
#defined bad/1.          % bad(B).          [B is bad transition]
#defined other/1.        % other(X).        [X is non-bad transition]
#defined sibling/2.      % sibling(G, S).   [S is sibling of good transition G]
#defined fixed/2.        % fixed(G, F).     [F cannot be relaxed in good transition G]
#defined source/3.       % source(X, F, V). [X is transition or bad; V=0 is false/eq, V=1 is true/gt]
#defined change/3.       % change(X, F, C). [X is transition, sibling, or bad; C is "dec", "inc", or "eqv"]
#defined yield/1.        % yield(Y).        [Y is a joint valuation for features that may result after a good transition]
#defined yield/2.        % yield(G, Y).     [Y is a joint valuation for features that may result after good transition G]
#defined value/3.        % value(Y, F, V).  [V=0 is false/eq, V=2 is true/gt]
#defined goal/1.         % goal(Y)          [Y is goal yield]

% TODO: yield/2 currently unused. It could be used to reduce grounding by removing yields that are provably impossible.
% INFO: #yields is 2**n where n is number of features


% ==== Transitions

transition(G) :- good(G).
transition(X) :- other(X).


% ==== Decision variables ====

#defined condition/3.    % condition(G, F, D). [D=0 is don't care, D=1 is value of F in source of G]
#defined effect/3.       % effect(G, F, D).    [D=0 is unknown, D=1 is change of F across G]

{ condition(G, F, 0..1) } = 1 :- good(G), feature(F), not fixed(G, F).
{ effect(G, F, 0..1) } = 1    :- good(G), feature(F), not fixed(G, F).
condition(G, F, 1)            :- fixed(G, F).
effect(G, F, 1)               :- fixed(G, F).

% Equivalent transitions and equivalence classes

eq(G, G)  :- good(G).
eq(G, G2) :- good(G), good(G2), G < G2,
             condition(G2, F, D) : condition(G, F, D);
             condition(G, F, D) : condition(G2, F, D);
             effect(G2, F, D) : effect(G, F, D);
             effect(G, F, D) : effect(G2, F, D);
             source(G2, F, V) : condition(G, F, 1), source(G, F, V);
             change(G2, F, C) : effect(G, F, 1), change(G, F, C).

class(G)  :- good(G), not eq(G2, G) : good(G2), G2 < G.


% ==== Consistent and compatible transitions ====

% Good transition G is CONSISTENT with transition X if their source states are consistent (non-symmetric relation)
consistent(G, B) :- good(G),        bad(B),  source(B, F, V) : feature(F), condition(G, F, 1), source(G, F, V).
consistent(G, T) :- good(G), transition(T),  source(T, F, V) : feature(F), condition(G, F, 1), source(G, F, V).

% Good transition G is COMPATIBLE with transition/sibling X if features affected by G change in X according to G (non-symmetric relation)
%compatible(G, B) :- good(G),     bad(B), change(B, F, C) : feature(F), effect(G, F, 1), change(G, F, C).
%compatible(G, S) :- good(G), sibling(S), change(S, F, C) : feature(F), effect(G, F, 1), change(G, F, C).

compatible(G, B) :- good(G),     bad(B), compatible(G, B, F) : feature(F), effect(G, F, 1).
compatible(G, S) :- good(G), sibling(S), compatible(G, S, F) : feature(F), effect(G, F, 1).
sibling(S)       :- good(G), sibling(G, S).

compatible(G, B, F) :- good(G),     bad(B), feature(F), effect(G, F, 1), change(G, F, C), change(B, F, C).
compatible(G, B, F) :- good(G),     bad(B), boolean(F), effect(G, F, 1), change(G, F, "inc"), change(B, F, "eqv"), source(B, F, 1).
compatible(G, B, F) :- good(G),     bad(B), boolean(F), effect(G, F, 1), change(G, F, "dec"), change(B, F, "eqv"), source(B, F, 0).
compatible(G, S, F) :- good(G), sibling(S), feature(F), effect(G, F, 1), change(G, F, C), change(S, F, C).
compatible(G, S, F) :- good(G), sibling(S), boolean(F), effect(G, F, 1), change(G, F, "inc"), change(S, F, "eqv"), source(S, F, 1).
compatible(G, S, F) :- good(G), sibling(S), boolean(F), effect(G, F, 1), change(G, F, "dec"), change(S, F, "eqv"), source(S, F, 0).


% ==== Safeness ====

% Good transition G is SAFE if for each bad transition B, G is either inconsistent or incompatible with B

safe(G)    :- good(G), safe(G, B) : bad(B).
safe(G, B) :- good(G), bad(B), not consistent(G, B).
safe(G, B) :- good(G), bad(B), not compatible(G, B).


% ==== Soundness ====

% Weak soundness
% Good transition G is (weakly) SOUND if for each transition T that is consistent with G, there is a sibling of T that is COMPATIBLE with SOME good transition G'
sound(G)                          :- good(G), there_is_compatible_sibling(T) : transition(T), consistent(G, T), G != T.
there_is_compatible_sibling(T)    :- good(G), sibling(T, S), compatible(G, S).

%*
% Good transition G is SOUND if for each transition T that is CONSISTENT with G, there is a sibling S of T that is COMPATIBLE with G
sound(G)                          :- good(G), there_is_compatible_sibling(G, T) : transition(T), consistent(G, T), G != T.
there_is_compatible_sibling(G, T) :- good(G), sibling(T, S), compatible(G, S), G != T.
*%


% ==== Structural closure ====

% Yield Y is POSSIBLE for good transition G if it may be a result of the relaxation of G (i.e., if value for each feature F may be a result of relaxation)
possible(Y, G)    :- yield(Y), good(G), possible(Y, G, F) : feature(F).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 0).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "dec").
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "inc"), value(Y, F, 1).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "eqv"), value(Y, F, V), source(G, F, V).

% Yield Y is ACTIVE if it is possible for some good transition G
active(Y)         :- possible(Y, G).

% Good transition G is CLOSED if each of its valid yields is covered
closed(G)         :- good(G), covered(Y) : possible(Y, G).

% Yield Y is COVERED if it is a goal yield, or there is good transition G that triggers at Y
covered(Y)        :- goal(Y).
covered(Y)        :- yield(Y), triggers(G, Y).

% Good transition G TRIGGERS at yield Y if source of G is consistent with Y
triggers(G, Y)    :- good(G), yield(Y), value(Y, F, V) : feature(F), condition(G, F, 1), source(G, F, V).


% ==== Hard/soft optimization ====

% Hard optimization (SAT)

:- good(G), not safe(G).

% Soft optimization

#minimize { 1@50,     class, G     : class(G) }.
#maximize { 1@40,   closure, G     : class(G), closed(G) }.
#maximize { 1@30, soundness, G     : class(G), sound(G) }.
#maximize { 1@20,   unknown, G, F  : class(G), effect(G, F, 0) }.
#maximize { 1@10,  dontcare, G, F  : class(G), condition(G, F, 0) }.

% ==== Show ====

#show.
#show eq/2.
#show class/1.
#show dont_care(G, F)    : class(G), condition(G, F, 0).
#show unknown(G, F)      : class(G), effect(G, F, 0).
#show nonsound(G)        : class(G), not sound(G).
#show nonclosed(G)       : class(G), not closed(G).
#show nonsafe(G)         : class(G), not safe(G).
#show noncovered(Y)      : yield(Y), active(Y), not covered(Y).
#show nonactive(Y)       : yield(Y), not active(Y).

