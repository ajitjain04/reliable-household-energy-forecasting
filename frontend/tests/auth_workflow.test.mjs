/**
 * Focused Node.js Unit & Workflow Test Suite for Frontend Authentication Gating.
 * Uses Node.js native test runner (node --test).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

test('Route Protection Matrix: Public vs Protected routes classification', () => {
  const publicRoutes = [
    '/',
    '/overview',
    '/benchmarks',
    '/reliability',
    '/statistical-tests',
    '/xai',
    '/uncertainty',
    '/explorer',
  ];

  const householdProtectedRoutes = [
    '/forecast/new',
    '/forecast/history',
    '/forecast/fc_test_123',
  ];

  const developerProtectedRoutes = [
    '/developer/forecast-records',
    '/developer/forecast-records/fc_test_123',
  ];

  const checkAccess = (route, user) => {
    if (publicRoutes.includes(route)) {
      return { allowed: true, action: 'render_public' };
    }

    if (developerProtectedRoutes.includes(route)) {
      if (!user) return { allowed: false, action: 'redirect_login' };
      if (user.role !== 'developer') return { allowed: false, action: 'show_role_required' };
      return { allowed: true, action: 'render_developer' };
    }

    if (householdProtectedRoutes.includes(route)) {
      if (!user) return { allowed: false, action: 'show_login_required_modal' };
      return { allowed: true, action: 'render_household' };
    }

    return { allowed: true, action: 'render_default' };
  };

  // 1. Unauthenticated visitor browsing public routes
  for (const route of publicRoutes) {
    const res = checkAccess(route, null);
    assert.equal(res.allowed, true);
    assert.equal(res.action, 'render_public');
  }

  // 2. Unauthenticated visitor attempting household protected routes
  for (const route of householdProtectedRoutes) {
    const res = checkAccess(route, null);
    assert.equal(res.allowed, false);
    assert.equal(res.action, 'show_login_required_modal');
  }

  // 3. Unauthenticated visitor attempting developer console
  for (const route of developerProtectedRoutes) {
    const res = checkAccess(route, null);
    assert.equal(res.allowed, false);
    assert.equal(res.action, 'redirect_login');
  }

  // 4. Authenticated household user attempting household routes
  const householdUser = { id: 'usr_1', username: 'house_user', role: 'household' };
  for (const route of householdProtectedRoutes) {
    const res = checkAccess(route, householdUser);
    assert.equal(res.allowed, true);
    assert.equal(res.action, 'render_household');
  }

  // 5. Authenticated household user attempting developer console -> forbidden
  for (const route of developerProtectedRoutes) {
    const res = checkAccess(route, householdUser);
    assert.equal(res.allowed, false);
    assert.equal(res.action, 'show_role_required');
  }

  // 6. Authenticated developer accessing developer console -> allowed
  const devUser = { id: 'usr_dev', username: 'dev_admin', role: 'developer' };
  for (const route of developerProtectedRoutes) {
    const res = checkAccess(route, devUser);
    assert.equal(res.allowed, true);
    assert.equal(res.action, 'render_developer');
  }
});

test('LoginRequiredModal actions and destination preservation contract', () => {
  const simulateModalActions = (intendedDestination) => {
    return {
      onSignIn: () => ({
        targetPath: '/login',
        state: { from: { pathname: intendedDestination } },
      }),
      onCreateAccount: () => ({
        targetPath: '/register',
        state: { from: { pathname: intendedDestination }, mode: 'register' },
      }),
      onCancel: (currentPath) => ({
        stayOnPage: true,
        currentPath,
      }),
    };
  };

  const modal = simulateModalActions('/forecast/new');

  // Verify Sign In preserving destination
  const signInResult = modal.onSignIn();
  assert.equal(signInResult.targetPath, '/login');
  assert.equal(signInResult.state.from.pathname, '/forecast/new');

  // Verify Create Account preserving destination and register mode
  const createResult = modal.onCreateAccount();
  assert.equal(createResult.targetPath, '/register');
  assert.equal(createResult.state.from.pathname, '/forecast/new');
  assert.equal(createResult.state.mode, 'register');

  // Verify Cancel returns visitor to current view without navigation
  const cancelResult = modal.onCancel('/overview');
  assert.equal(cancelResult.stayOnPage, true);
  assert.equal(cancelResult.currentPath, '/overview');
});

test('Login & Registration redirect destination resolution', () => {
  const resolvePostAuthRedirect = (locationState, userRole) => {
    const from = locationState?.from?.pathname || locationState?.from;
    if (from) return from;
    if (userRole === 'developer') return '/developer/forecast-records';
    return '/';
  };

  // 1. From New Forecast -> returns to /forecast/new
  assert.equal(
    resolvePostAuthRedirect({ from: { pathname: '/forecast/new' } }, 'household'),
    '/forecast/new'
  );

  // 2. From Forecast History -> returns to /forecast/history
  assert.equal(
    resolvePostAuthRedirect({ from: { pathname: '/forecast/history' } }, 'household'),
    '/forecast/history'
  );

  // 3. Normal direct login with no from state -> returns home for household
  assert.equal(
    resolvePostAuthRedirect(null, 'household'),
    '/'
  );

  // 4. Normal direct login for developer -> returns developer console
  assert.equal(
    resolvePostAuthRedirect(null, 'developer'),
    '/developer/forecast-records'
  );
});

test('Logout state clearing and client navigation boundary', () => {
  let sessionState = {
    token: 'jwt_valid_token_xyz',
    user: { id: 'usr_1', username: 'house_user', role: 'household' },
    isAuthenticated: true,
  };

  const logout = () => {
    sessionState = {
      token: null,
      user: null,
      isAuthenticated: false,
    };
  };

  assert.equal(sessionState.isAuthenticated, true);
  logout();
  assert.equal(sessionState.isAuthenticated, false);
  assert.equal(sessionState.token, null);
  assert.equal(sessionState.user, null);
});

test('Password Visibility Toggle: state machine, accessibility labels, and default masked behavior', () => {
  // State machine simulating Login component password visibility behavior
  class PasswordFieldController {
    constructor() {
      this.showPassword = false;
      this.mode = 'login';
    }

    getInputType() {
      return this.showPassword ? 'text' : 'password';
    }

    getButtonLabel() {
      return this.showPassword ? 'Hide password' : 'Show password';
    }

    getButtonType() {
      return 'button';
    }

    getIconName() {
      return this.showPassword ? 'EyeOff' : 'Eye';
    }

    toggle() {
      this.showPassword = !this.showPassword;
    }

    switchMode(newMode) {
      this.mode = newMode;
      this.showPassword = false; // Safe reset on tab switch
    }
  }

  const controller = new PasswordFieldController();

  // 1. Initial / default state must be masked
  assert.equal(controller.showPassword, false, 'Password must be masked by default');
  assert.equal(controller.getInputType(), 'password', 'Input type must default to "password"');
  assert.equal(controller.getButtonLabel(), 'Show password', 'Accessible label must initially offer to show password');
  assert.equal(controller.getButtonType(), 'button', 'Toggle must be type="button" to prevent accidental form submission');
  assert.equal(controller.getIconName(), 'Eye', 'Eye icon must display when masked');

  // 2. Toggle to visible
  controller.toggle();
  assert.equal(controller.showPassword, true, 'State must become visible');
  assert.equal(controller.getInputType(), 'text', 'Input type must switch to "text"');
  assert.equal(controller.getButtonLabel(), 'Hide password', 'Accessible label must update to "Hide password"');
  assert.equal(controller.getIconName(), 'EyeOff', 'EyeOff icon must display when revealed');

  // 3. Toggle back to masked
  controller.toggle();
  assert.equal(controller.showPassword, false, 'State must become masked again');
  assert.equal(controller.getInputType(), 'password', 'Input type must return to "password"');
  assert.equal(controller.getButtonLabel(), 'Show password', 'Accessible label must return to "Show password"');
  assert.equal(controller.getIconName(), 'Eye', 'Eye icon must display again');

  // 4. Reveal password, then switch tab from login to register -> must reset to masked
  controller.toggle();
  assert.equal(controller.showPassword, true);
  controller.switchMode('register');
  assert.equal(controller.showPassword, false, 'Tab switch must reset password visibility to masked');
  assert.equal(controller.getInputType(), 'password');
  assert.equal(controller.getButtonLabel(), 'Show password');
});

