import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {test} from 'node:test';
import {createSparkrunTool} from '../tool.mjs';

function fixtureTool(script) {
    return createSparkrunTool((file, args, options, callback) => {
        assert.equal(file, 'sparkrun');
        assert.equal(options.shell, false);
        return execFile(process.execPath, ['-e', script, '--', ...args], options, callback);
    });
}

test('SDK callback accepts call ID first; preserves argument boundaries and stderr', async () => {
    const tool = fixtureTool('console.log(JSON.stringify(process.argv.slice(1))); console.error("warning");');
    const args = ['run', 'recipe with spaces', '-o', 'value=$(echo forbidden); `echo no`', '--no-follow'];
    const result = await tool.execute('call-123', {args});
    assert.equal(result.isError, false);
    assert.equal(result.details.exitCode, 0);
    assert.deepEqual(JSON.parse(result.content[0].text.split('\n')[0]), args);
    assert.match(result.content[0].text, /warning/);
});

test('failed commands retain output and exit status without echoing secret args', async () => {
    const result = await fixtureTool('console.log("partial"); console.error("bad request"); process.exit(7);')
        .execute('id', {args: ['proxy', 'start', '--master-key', 'test-secret']});
    assert.equal(result.isError, true);
    assert.equal(result.details.exitCode, 7);
    assert.match(result.content[0].text, /partial/);
    assert.match(result.content[0].text, /bad request/);
    assert.doesNotMatch(result.content[0].text, /test-secret/);
});

test('invalid inputs never start a process, including obsolete command-string input', async () => {
    const tool = createSparkrunTool(() => { throw new Error('must not execute'); });
    for (const params of [undefined, {command: 'sparkrun list'}, {args: []}, {args: [42]},
        {args: ['bad\0arg']}, {args: ['sparkrun', 'list']}, {args: ['list'], timeout: 0},
        {args: ['list'], timeout: 3_600_001}, {args: ['list'], timeout: 1.5}]) {
        const result = await tool.execute('id', params);
        assert.equal(result.isError, true);
        assert.doesNotMatch(result.content[0].text, /must not execute/);
    }
});

test('stdin closes instead of hanging at an unattended prompt', async () => {
    const result = await fixtureTool('process.stdin.resume(); process.stdin.on("end", () => console.log("eof"));')
        .execute('id', {args: ['setup'], timeout: 5000});
    assert.equal(result.isError, false);
    assert.match(result.content[0].text, /eof/);
});

test('timeouts report remote workload uncertainty', async () => {
    const result = await fixtureTool('setInterval(() => {}, 1000);')
        .execute('id', {args: ['run', 'recipe'], timeout: 100});
    assert.equal(result.isError, true);
    assert.equal(result.details.timedOut, true);
    assert.match(result.content[0].text, /Remote workloads may still be running/);
});

test('SDK cancellation reaches the running child', async () => {
    const controller = new AbortController();
    const pending = fixtureTool('setInterval(() => {}, 1000);')
        .execute('id', {args: ['run', 'recipe']}, controller.signal);
    controller.abort();
    const result = await pending;
    assert.equal(result.isError, true);
    assert.equal(result.details.cancelled, true);
    assert.equal(result.details.timedOut, false);
});

test('output limits are errors, not success or timeouts', async () => {
    const result = await fixtureTool('console.log("x".repeat(5 * 1024 * 1024));')
        .execute('id', {args: ['logs', 'workload']});
    assert.equal(result.isError, true);
    assert.equal(result.details.truncated, true);
    assert.equal(result.details.timedOut, false);
});

test('missing executable gives an actionable error', async () => {
    const tool = createSparkrunTool((_file, args, options, callback) =>
        execFile('sparkrun-test-executable-does-not-exist', args, options, callback));
    const result = await tool.execute('id', {args: ['status']});
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /not found.*PATH/);
});
